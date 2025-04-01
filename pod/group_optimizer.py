# Standard library imports
import os
import time
from collections import deque, defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import List, Optional, Literal, Union
from io import BytesIO

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from torchvision.transforms.functional import resize
from tqdm import tqdm
import kornia
from torchvision.transforms import ToTensor
import roma

# Project-specific imports
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import Viewer, VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.model_components.losses import depth_ranking_loss
from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipeline
import viser.transforms as vtf
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
from pod.utils.utils import *
from pod.utils.ff_utils import *
from pod.utils.geo_utils import *

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))

def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def get_vid_frame(cap,timestamp):
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number based on the timestamp and fps
    frame_number = min(int(timestamp * fps),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1))
    print(frame_number)
    # Set the video position to the calculated frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    success, frame = cap.read()
    # convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def cauchy_loss(x:torch.Tensor, y:torch.Tensor, scale:float = 1.0):
    """
    Cauchy loss between x and y
    """
    return torch.log(1 + ((x - y) / scale) ** 2).mean()

def mnn_matcher(feat_a, feat_b):
    """
    feat_a: NxD
    feat_b: MxD
    return: K, K (indices in feat_a and feat_b)
    """
    device = feat_a.device
    sim = feat_a.mm(feat_b.t())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    return ids1[mask], nn12[mask]

@torch.no_grad
def get_scale_and_shift(x:torch.Tensor):
    shift = x.median()
    devs = (x - shift).abs()#median deviation is the scale
    return devs.median(),shift 

class Joint_RigidGroupOptimizer:
    use_depth: bool = True
    rank_loss_mult: float = 0.1
    mask_loss_mult: float = 1
    rank_loss_erode: int = 3
    blur_kernel_size: int = 5

    def __init__(
        self,
        pipeline: GarfieldGaussianPipeline,
        init_c2w: Cameras, 
        render_lock = nullcontext(), 
        device = 'cuda',
        move_object_to_worldcenter: bool = True,
        ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        self.device = device 
        self.pipeline = pipeline
        self.dig_model = pipeline.model
        #detach all the params to avoid retain_graph issue
        self.dig_model.gauss_params['means'] = self.dig_model.gauss_params['means'].detach().clone()
        self.dig_model.gauss_params['quats'] = self.dig_model.gauss_params['quats'].detach().clone()
        self.dino_loader = pipeline.datamanager.dino_dataloader
        self.group_masks = pipeline.cluster_labels.to(self.device)
        self.input_group_labels= self.pipeline.cluster_labels.int().to(self.device)  
        self.input_group_masks = [(cid == self.input_group_labels).to(self.device)  for cid in range(self.input_group_labels.max() + 1)]
        self.group_labels = pipeline.cluster_labels.int().to(self.device)
        self.init_c2w = deepcopy(init_c2w).to(self.device)
        #store a 7-vec of trans, rotation for each group
        self.pose_deltas = torch.zeros(int(self.group_masks.unique().max()+1),7,dtype=torch.float32,device=self.device)
        self.pose_deltas[:,3:] = torch.tensor([1,0,0,0],dtype=torch.float32,device=self.device)
        self.pose_deltas = torch.nn.Parameter(self.pose_deltas)
       
        self.init_means = self.dig_model.gauss_params['means'].detach().clone()
        self.init_quats = self.dig_model.gauss_params['quats'].detach().clone()
       
       
        self.keyframes = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock

        #joint masks
        self.root_label = pipeline.root_label   
        self.joint_position = pipeline.curr_joints
        self.tree = pipeline.tree
        self.parent_map = pipeline.parent_map    
        self.init_joint_position = self.joint_position.clone()
        self.num_joints = len(self.joint_position)
        
        # fast calculation of the relative joint positions
        self.w_poses = torch.tile(torch.eye(4),(self.dig_model.num_points,1,1)).to(self.device)
        
        self.garfield_old_parent_map = pipeline.parent_map
        self.garfield_old_tree = pipeline.tree
        self.garfield_old_joint_position = self.joint_position.clone()
        
        self.is_initialized = False
        self.relative_joint_poses = self.relative_joint_positions(self.joint_position)
        self.local_gaussian_pose = self.gaussian_local_pose()
        
        #Justin's blur
        #self.blur = torchvision.transforms.GaussianBlur(kernel_size=[21,21]).cuda()
        k = self.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.alpha = 0.25
        self.gamma = 2.0
        
        #initialize the mask
        self.first_frame_mask = None

        #move object to world center
        if move_object_to_worldcenter:
            self.old_object_center = self.init_means.mean(dim=0).clone()
            self.move_object_to_worldcenter()
    
    def move_object_to_worldcenter(self):
        init_object_center = self.init_means.mean(dim=0)
        if torch.allclose(init_object_center, torch.tensor([0., 0., 0.], device=init_object_center.device), atol=1e-7):
            print("Object is already in the world center")
        else:
            print("Object is not in the world center")
            #move the object to the world center
            self.init_means -= init_object_center
            self.joint_position -= init_object_center
            self.relative_joint_poses = self.relative_joint_positions(self.joint_position)
            self.local_gaussian_pose = self.gaussian_local_pose()
            self.pipeline.model.gauss_params['means'] = self.init_means.clone()
            self.pipeline.connected_points -= init_object_center.detach().clone().cpu()
            self.reset_transforms()

    def reset_pose_deltas(self):
        with torch.no_grad():
            self.pose_deltas[:, 3:] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
            self.pose_deltas[:, :3] = 0
            self.apply_to_model(self.pose_deltas)
        self.pose_deltas = torch.nn.Parameter(self.pose_deltas)

    #we can use one level tree structure
    def reset_tree_structure(self,tree = None):
        #root part (1)(no gs) -> other parts(n)
        self.reset_transforms()
        if tree is not None:
            self.tree = tree
        else:
            self.root_label = int(self.group_masks.unique().max().item()+1)
            object_part_tree = {self.root_label: [i for i in range(self.num_joints)]}
            self.tree = object_part_tree
           
        for i,g in enumerate(self.input_group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            if i!=self.root_label:
                self.joint_position[i] = gp_centroid

        self.joint_position = torch.cat((self.joint_position,self.init_means.mean(dim=0).unsqueeze(0)),dim=0)
        self.num_joints = len(self.joint_position)
        
        self.relative_joint_poses = self.relative_joint_positions(self.joint_position)
        self.local_gaussian_pose = self.gaussian_local_pose()
        self.parent_map = build_parent_mapping(self.tree)

        self.pose_deltas = torch.zeros(int(self.group_masks.unique().max()+2),7,dtype=torch.float32,device=self.device)
        self.pose_deltas[:,3:] = torch.tensor([1,0,0,0],dtype=torch.float32,device=self.device)
        self.pose_deltas = torch.nn.Parameter(self.pose_deltas)
        print("reset_tree_structure", self.tree)
    
    def gaussian_local_pose(self):
        """
        Returns the local pose of each gaussian in the camera coordinate system
        """
        # V_wg = T_wc V_cg -> V_cg = T_wc^-1 V_wg : gaussian pose in child frame
        updated_curr_means = self.init_means.clone()
        updated_curr_rotmats = quat_to_rotmat(self.init_quats.clone())
        
        local_transoform = torch.tile(torch.eye(4),(len(self.group_masks),1,1)).to(self.device)
        for node in self.relative_joint_poses.keys():
            group_inds = np.isin(self.group_masks.detach().cpu(),[node])
            group_inds = torch.tensor(group_inds).to(self.device)
           
            curr_means = updated_curr_means[group_inds]
            curr_rotates = updated_curr_rotmats[group_inds]
            
            w_vector = torch.tile(torch.eye(4),(len(curr_means),1,1)).to(self.device)
            w_vector[:,:3,:3] = curr_rotates
            w_vector[:,:3,3] = curr_means
            
            w_frame = torch.eye(4).to(self.device)
            w_frame[:3,3] = self.joint_position[node]
            
            local_transoform[group_inds] = torch.matmul(torch.inverse(w_frame)[None].repeat(w_vector.shape[0],1,1),w_vector)
        
        return local_transoform

    def relative_joint_positions(self,joints):
        """
        input: world position of the joints 
        Returns the relative joint positions given the joint positions 4*4
        """
        root_transformation = torch.eye(4,device=self.device)
        root_transformation[:3,3] = joints[self.root_label]
        
        relative_joints = {self.root_label: root_transformation}
        for p_node,c_node in self.tree.items():
           for node in c_node:
               relative_joints[node] = torch.eye(4,device=self.device)
               relative_joints[node][:3,3] = joints[node] - joints[p_node]
        
        return relative_joints
                   
    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params["means"] = self.init_means.detach().clone()
            self.dig_model.gauss_params["quats"] = self.init_quats.detach().clone()

    def initialize_camera_pose(self, niter=200, n_seeds=6, render=False, lr=0.01):
        renders = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj,niter,lr =0.01):
            "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
            # an seperate optimization function for the initial pose
            self.reset_transforms()          
            whole_pose_adj = start_pose_adj.detach().clone()
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj], lr=lr)
            scheduler = ExponentialDecayScheduler(
                        ExponentialDecaySchedulerConfig(
                            lr_final= lr * 0.5, max_steps=niter,
                        )
            ).get_scheduler(optimizer, lr)
            
            for _ in range(niter):
                optimizer.zero_grad()
                with self.render_lock:
                    self.dig_model.eval()
                    self.apply_to_model(whole_pose_adj)
                    dig_outputs = self.dig_model.get_outputs(self.init_c2w)
                    if "dino" not in dig_outputs:
                        return None, None, None
                init_blur = torchvision.transforms.GaussianBlur(kernel_size=[21,21]).cuda()
                blur_dino_feats = (
                    init_blur(dig_outputs["dino"].permute(2, 0, 1)[None])
                    .squeeze()
                    .permute(1, 2, 0)
                )
                
                dino_feats = dig_outputs["dino"]
                frame_pca_feats = self.first_frame_pca_feats
                pix_loss = frame_pca_feats - dino_feats
                blurred_pix_loss = frame_pca_feats - blur_dino_feats
                          
                dino_loss = 0.1*pix_loss.abs().mean() + 1.0*blurred_pix_loss.abs().mean()
                loss = dino_loss

                #add mask loss
                if self.first_frame_mask is not None:
                    mask_mse_loss = torch.square(self.first_frame_mask - dig_outputs['accumulation'][...,0])
                    if self.first_frame_hand_mask is not None:
                        mask_mse_loss = mask_mse_loss[self.first_frame_hand_mask]
                    
                    mask_mse_loss = mask_mse_loss.mean()
                    loss += self.mask_loss_mult * mask_mse_loss
                
                loss.backward()
                indices = torch.arange(whole_pose_adj.size(0))  # Create a range of indices
                mask = (indices == self.root_label).float()  # Compare indices to `self.root_label` and convert to float
                if whole_pose_adj.grad is not None:  # Check if gradients exist
                    expanded_mask = mask.unsqueeze(1)  # Example adjustment, tailor to your needs
                    expanded_mask = expanded_mask.expand_as(whole_pose_adj.grad).to(self.device)
                whole_pose_adj.grad *= expanded_mask
                optimizer.step()
                scheduler.step()
                
                if render:
                    renders.append(dig_outputs["rgb"].detach())
                    
            self.is_initialized = True
            return dig_outputs, loss, whole_pose_adj.data.detach().clone()

        def find_pixel(n_gauss=10000):
            """
            returns the y,x coord and box size of the object in the video frame, based on the dino features
            and mutual nearest neighbors
            """
            samps = torch.randint(
                0, self.dig_model.num_points, (n_gauss,), device=self.device
            )
            nn_inputs = self.dig_model.gauss_params["dino_feats"][samps]
            # dino_feats = self.dig_model.nn(nn_inputs.half()).float()  # NxC
            dino_feats = self.dig_model.nn(nn_inputs)  # NxC
            downsamp_factor = 4
            downsamp_frame_feats = self.first_frame_pca_feats[
                ::downsamp_factor, ::downsamp_factor, :
            ]
            frame_feats = downsamp_frame_feats.reshape(
                -1, downsamp_frame_feats.shape[-1]
            )  # (H*W) x C
            _, match_ids = mnn_matcher(dino_feats, frame_feats)
            x, y = (match_ids % (self.init_c2w.width / downsamp_factor)).float(), (
                match_ids // (self.init_c2w.width / downsamp_factor)
            ).float()
            x, y = x * downsamp_factor, y * downsamp_factor
            return y, x, torch.tensor([y.mean().item(), x.mean().item()], device=self.device)


        best_loss = float("inf")
        ys, xs, best_pix = find_pixel()  # Find the best pixel to start frame
        ray = self.init_c2w.generate_rays(0, best_pix)
        dist = 1.0
        point = ray.origins + ray.directions * dist #object start point
        
        for z_rot in tqdm(np.linspace(0, np.pi * 2, n_seeds)):
            whole_pose_adj = torch.zeros(int(self.group_masks.unique().max()+1), 7, dtype=torch.float32, device=self.device)
            whole_pose_adj[:, 3:] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
            # x y z qw qx qy qz
            # (x,y,z) = something along ray - centroid
            quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).to(self.device)
            
            whole_pose_adj[self.root_label, :3] = point - self.relative_joint_poses[self.root_label][:3,3].clone()
            whole_pose_adj[self.root_label, 3:] = quat
            dig_outputs, loss, final_pose = try_opt(whole_pose_adj,niter,lr)

            if loss is not None and loss < best_loss:
                best_loss = loss.detach().clone()
                best_outputs = {k:v.detach().clone() for k,v in dig_outputs.items()}
                best_pose = final_pose.detach().clone()
                
        
        if loss is not None and loss < best_loss:
            best_loss = loss.detach().clone()
            #best_outputs is a dict,detach it to avoid retain_graph issue
            best_outputs = {k:v.detach().clone() for k,v in dig_outputs.items()}
            best_pose = final_pose.detach().clone()
                
        
        self.reset_transforms()        

        with torch.no_grad():
            ###############################################################################################################
            #instead of applying the best pose, we apply the identity it to the camera  
            #T_cw @ T_wp = T_cp
            #T_cw @ T_wp' = T_cp'
            #T_cw' @ T_wp = T_cp'
            
            #T_cw @ A @ T_wp = T_cp'
            #A = T_wp' @ T_wp^-1
            #T_cw' = T_cw @ A
            #T_cw' = T_cw @ T_wp' @ T_wp^-1
            #T_wc' = T_wp @ T_wp'^-1 @ T_wc
            T_wp_initial = self.relative_joint_poses[self.root_label].clone()           
            T_wp_updated = self.relative_joint_poses[self.root_label].clone() @ pose2mat(best_pose).to(self.device)[self.root_label] 
            #apply the inverse of the translation to the init_camera
            best_init_c2w = deepcopy(self.init_c2w.camera_to_worlds)
            bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32,device=self.device)  # Bottom row for homogeneous coordinates
            T_best_wc = torch.cat((best_init_c2w.squeeze(0), bottom), dim=0)  # Form the full 4x4 matrix
            T_best_wc_updated = T_wp_initial @ torch.linalg.inv(T_wp_updated) @ T_best_wc 

            T_best_cw_updated = torch.linalg.inv(T_best_wc_updated)
            T_best_cp_updated = T_best_cw_updated @ T_wp_initial
            T_best_cp_initial = torch.linalg.inv(T_best_wc) @ T_wp_updated
            print(T_best_cp_initial )
            print(T_best_cp_updated )
            # debug
            best_init_c2w_updated =  T_best_wc_updated [:3,:].unsqueeze(0)                
            self.init_c2w.camera_to_worlds = best_init_c2w_updated
              
            self.pipeline.eval()
            self.pipeline.model.set_background(torch.ones(3))
            outputs = self.dig_model.get_outputs(self.init_c2w)
            ###############################################################################################################    
                
            with torch.no_grad():
                self.pose_deltas[:, 3:] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
                self.pose_deltas[:, :3] = 0
                
        with torch.no_grad():
            target_frame_rgb = self.first_frame_rgb
            image_array = np.zeros((self.init_c2w.height, self.init_c2w.width*3, 3), dtype=np.uint8)
            image_array[:, :self.init_c2w.width] = (outputs["rgb"].detach().cpu().numpy() * 255).astype(np.uint8)
            image_array[:, self.init_c2w.width:self.init_c2w.width*2] = (target_frame_rgb.cpu().numpy() * 255).astype(np.uint8)
            image_array[:, self.init_c2w.width*2:self.init_c2w.width*3] = ((outputs["rgb"]*0.5+target_frame_rgb*0.5).detach().abs().cpu().numpy() * 255).astype(np.uint8)
            
        return image_array
   
    def render_pose(self,video_idx, pose_deltas):
        with torch.no_grad():
            with self.render_lock:
                self.dig_model.eval()
                # apply_model_time = time.time()
                self.apply_to_model(pose_deltas,None)
                video_init_cam = self.multi_video_data[video_idx]['init_c2w_matrix']
                h,w = video_init_cam['height'],video_init_cam['width']
                cx,cy = video_init_cam['cx'],video_init_cam['cy']
                fx,fy = video_init_cam['fx'],video_init_cam['fy']
                cam_pose = video_init_cam['camera_to_worlds']
                init_cam = Cameras(camera_to_worlds=cam_pose, fx = fx, fy = fy, cx = cx, cy = cy, width=w, height=h).to(self.device)
                self.init_c2w = init_cam
                dig_outputs = self.dig_model.get_outputs(self.init_c2w)
                # print("apply_model_time",time.time()-apply_model_time)
                #load the camera 
        return dig_outputs['rgb'].detach().cpu().numpy()      
    
    def get_rgb_render(self,video_idx: int,pose_deltas, axis = None):
        with self.render_lock:
            self.dig_model.eval()
            self.apply_to_model(pose_deltas,axis)
            with torch.no_grad():
                video_init_cam = self.multi_video_data[video_idx]['init_c2w_matrix']
                h,w = video_init_cam['height'],video_init_cam['width']
                cx,cy = video_init_cam['cx'],video_init_cam['cy']
                fx,fy = video_init_cam['fx'],video_init_cam['fy']
                cam_pose = video_init_cam['camera_to_worlds']
                init_cam = Cameras(camera_to_worlds=cam_pose, fx = fx, fy = fy, cx = cx, cy = cy, width=w, height=h).to(self.device)
                self.init_c2w = init_cam
            
            dig_outputs = self.dig_model.get_outputs(self.init_c2w)
        rendered_rgb = dig_outputs['rgb'].detach().cpu().numpy()        
        return rendered_rgb
        
    def loss(self,
             video_idx: int ,
             frame_idx: int, 
             pose_deltas, 
             input_camera_pose = None,
             axis=None, 
             use_depth = True, 
             use_rgb = False ,
             use_mask = True, 
             mask_hands = False,
             use_part_merging = False,
             ):
        
        loss = {}
        
        #load the camera 
        with torch.no_grad():
            if input_camera_pose is not None:
                self.init_c2w = input_camera_pose
            else:
                video_init_cam = self.multi_video_data[video_idx]['init_c2w_matrix']
                h,w = video_init_cam['height'],video_init_cam['width']
                cx,cy = video_init_cam['cx'],video_init_cam['cy']
                fx,fy = video_init_cam['fx'],video_init_cam['fy']
                cam_pose = video_init_cam['camera_to_worlds']
                init_cam = Cameras(camera_to_worlds=cam_pose, fx = fx, fy = fy, cx = cx, cy = cy, width=w, height=h).to(self.device)
                self.init_c2w = init_cam

        #load the idx-th frame
        srgb_frame = self.multi_video_data[video_idx]['rgb'][frame_idx].to(self.device)
        sframe_depth = self.multi_video_data[video_idx]['depth'][frame_idx].to(self.device)
        sframe_pca_feats = self.multi_video_data[video_idx]['dino_pca_feats'][frame_idx].to(self.device)
        sframe_pca_feats = resize(sframe_pca_feats.permute(2,0,1), (self.init_c2w.height,self.init_c2w.width)).permute(1,2,0).contiguous()
        if mask_hands:
            shand_mask = self.hand_mask_lists[video_idx][frame_idx].to(self.device)
            shand_mask [self.mask_lists[video_idx][frame_idx].to(self.device) == 1] = 1
            mean_depth =  sframe_depth[self.mask_lists[video_idx][frame_idx].to(self.device) == 1].mean()
            shand_mask[(sframe_depth<mean_depth).squeeze(-1)] = 1


        #load the idx-th camera
        with self.render_lock:
            self.pipeline.eval()
            self.pipeline.model.set_background(torch.ones(3))
            self.apply_to_model(pose_deltas,axis)
            dig_outputs = self.dig_model.get_outputs(self.init_c2w)
        
        rendered_rgb = dig_outputs['rgb']      
        
        if 'dino' not in dig_outputs:
            self.reset_transforms()
            raise RuntimeError("Lost tracking")
        
        #use object mask
        with torch.no_grad():
            object_mask = dig_outputs["accumulation"] > 0.8
        
        if not object_mask.any():
            # print("No valid masks")
            return None, rendered_rgb

        blur_dino_feats = (
            self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
            .squeeze()
            .permute(1, 2, 0)
        )
        
        dino_feats = dig_outputs["dino"]
        blur_dino_feats = torch.where(object_mask, dino_feats, blur_dino_feats)
        
        
        pix_loss = sframe_pca_feats - dino_feats
        blurred_pix_loss = sframe_pca_feats - blur_dino_feats
        if mask_hands:
            pix_loss = pix_loss[shand_mask]#.norm(dim=-1)
            blurred_pix_loss = blurred_pix_loss[shand_mask]#.norm(dim=-1)
        
        dino_loss = 1.0*pix_loss.abs().mean() + 1.0*blurred_pix_loss.abs().mean()
        loss['dino_loss'] = dino_loss
        
        if use_mask:
            obj_mask = self.mask_lists[video_idx][frame_idx].to(self.device).to(torch.float32)
            mask_mse_loss = torch.square(obj_mask - dig_outputs['accumulation'][...,0])
            if mask_hands:
                mask_mse_loss = mask_mse_loss[shand_mask]
            mask_mse_loss = mask_mse_loss.mean()
            loss['mask_loss'] = mask_mse_loss 

            #mx : I am not sure about this loss
            pix_loss_masked = sframe_pca_feats*obj_mask[...,None] - dino_feats*object_mask.to(torch.float32)
            blurred_pix_loss_masked = sframe_pca_feats*obj_mask[...,None] - blur_dino_feats*object_mask.to(torch.float32)
            if mask_hands:
                pix_loss_masked = pix_loss_masked[shand_mask]
                blurred_pix_loss_masked = blurred_pix_loss_masked[shand_mask]
            dino_loss_masked = 1.0*pix_loss_masked.abs().mean() + 1.0*blurred_pix_loss_masked.abs().mean()
            loss['dino_loss_masked'] = dino_loss_masked
        
        else:
            loss['mask_loss'] = 0.
        
        if use_depth and self.use_depth:
           # This is ranking loss for monodepth (which is disparity)
            frame_depth = 1.0 / sframe_depth # convert disparity to depth
            N = 10000
            # erode the mask by like 10 pixels
            object_mask = dig_outputs['accumulation']>.9
            object_mask = object_mask & (~frame_depth.isnan())
            # commenting this out for now since it sometimes crashes with no valid pixels
            object_mask = kornia.morphology.erosion(
                object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device=self.device)
            ).squeeze().bool()
            if mask_hands:
                object_mask = object_mask & shand_mask
            valid_ids = torch.where(object_mask)
           
            if len(valid_ids[0]) > 0:
                rand_samples = torch.randint(
                    0, valid_ids[0].shape[0], (N,), device= self.device
                )
                rand_samples = (
                    valid_ids[0][rand_samples],
                    valid_ids[1][rand_samples],
                )
                rend_samples = dig_outputs["depth"][rand_samples]
                mono_samples = frame_depth[rand_samples]
                rank_loss = depth_ranking_loss(rend_samples, mono_samples)
                loss['depth_loss'] = self.rank_loss_mult * rank_loss
 
            else:
                loss['depth_loss'] = 0

        if use_rgb:
            rgb_loss =  0.05 *(dig_outputs['rgb']-srgb_frame)#.abs().mean()
            if mask_hands:
                rgb_loss = rgb_loss[shand_mask]
            rgb_loss = rgb_loss.abs().mean()
            loss['rgb_loss'] = rgb_loss
                

        if use_part_merging:

            connected_graph = self.pipeline.connected_graph
            if pose_deltas.shape[0] != len(connected_graph):
                part_pose_deltas = pose_deltas[:-1]
            else:
                part_pose_deltas = pose_deltas
            part_pose_deltas_t = part_pose_deltas[:,:3] 
            part_pose_deltas_r = quaternion_to_rotation_matrix(part_pose_deltas[:,3:])
            part_pose_deltas_transformation = translation_rotation_to_4x4_matrix(part_pose_deltas_t,part_pose_deltas_r)
            
            loss['part_merging_loss'] = 0

            indices = torch.nonzero(connected_graph, as_tuple=False)  # (N, 2)
            indices = indices[indices[:, 0] < indices[:, 1]] 
            i_indices, j_indices = indices[:, 0], indices[:, 1] 
            connected_points = self.pipeline.connected_points[i_indices, j_indices].clone().to(self.device)
            T_rp = translation_rotation_to_4x4_matrix(connected_points,torch.eye(3, device=self.device).expand(len(indices), 3, 3))  # Shape: (N, 4, 4)

            T_ri = torch.stack([self.relative_joint_poses[i].clone() for i in i_indices.tolist()])# Shape: (N, 4, 4)
            T_rj = torch.stack([self.relative_joint_poses[j].clone() for j in j_indices.tolist()])

            # Compute T_ip and T_jp
            T_ip = torch.linalg.inv(T_ri) @ T_rp  # (N, 4, 4)
            T_jp = torch.linalg.inv(T_rj) @ T_rp

            # Compute transformed poses
            T_ri_prime = T_ri @ part_pose_deltas_transformation[i_indices]  # (N, 4, 4)
            T_rj_prime = T_rj @ part_pose_deltas_transformation[j_indices]

            # Compute final transformations
            T_rp_prime1 = T_ri_prime @ T_ip  # (N, 4, 4)
            T_rp_prime2 = T_rj_prime @ T_jp

            # Compute distance metric in batch
            loss['part_merging_loss'] = transformation_matrix_distance(T_rp_prime1, T_rp_prime2, "geodesic", t_weight=10).sum()


        return loss,rendered_rgb
    

    def loss_calculation(
             self,
             input_pose_deltas,
             input_dig_ouputs, #'rgb'/ 'dino'/ 'accumulation'/ 'depth'
             groundtruth, # rgb/dino/depth/object/hand_mask/
             use_depth = True, 
             use_rgb = False ,
             use_mask = True, 
             mask_hands = True,
             use_part_merging = False,
             ):
        
        loss = {}
        dig_outputs = input_dig_ouputs
        pose_deltas = input_pose_deltas
        
    
        #load the idx-th frame
        srgb_frame = groundtruth['rgb'].to(self.device)
        sframe_depth = groundtruth['depth'].to(self.device)
        sframe_pca_feats = groundtruth['dino_pca_feats'].to(self.device)
        if mask_hands:
            shand_mask = groundtruth['hand_mask'].to(self.device)
            shand_mask [ groundtruth['object_mask'] == 1] = 1
            mean_depth = sframe_depth[ groundtruth['object_mask'] == 1].mean()
            shand_mask[(sframe_depth<mean_depth).squeeze(-1)] = 1


        rendered_rgb = dig_outputs['rgb'].detach().cpu().numpy()      
        
        if 'dino' not in dig_outputs:
            self.reset_transforms()
            raise RuntimeError("Lost tracking")
        
        #use object mask
        with torch.no_grad():
            object_mask = dig_outputs["accumulation"] > 0.8
        
        if not object_mask.any():
            # print("No valid masks")
            return None, rendered_rgb

        blur_dino_feats = (
            self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
            .squeeze()
            .permute(1, 2, 0)
        )
        
        dino_feats = dig_outputs["dino"]
        assert dino_feats.shape == sframe_pca_feats.shape, f"{dino_feats.shape} != {sframe_pca_feats.shape}"+ str(self.pipeline.model.training)
        blur_dino_feats = torch.where(object_mask, dino_feats, blur_dino_feats)
        
        
        pix_loss = sframe_pca_feats - dino_feats
        blurred_pix_loss = sframe_pca_feats - blur_dino_feats
        
        if mask_hands:
            pix_loss = pix_loss[shand_mask]#.norm(dim=-1)
            blurred_pix_loss = blurred_pix_loss[shand_mask]#.norm(dim=-1)
        
        dino_loss = 1.0*pix_loss.abs().mean() + 1.0*blurred_pix_loss.abs().mean()
        loss['dino_loss'] = dino_loss
        
        if use_mask:
            obj_mask = groundtruth['object_mask'].to(self.device).to(torch.float32)
            mask_mse_loss = torch.square(obj_mask - dig_outputs['accumulation'][...,0])
            if mask_hands:
                mask_mse_loss = mask_mse_loss[shand_mask]
            mask_mse_loss = mask_mse_loss.mean()
            loss['mask_loss'] = mask_mse_loss 

            pix_loss_masked = sframe_pca_feats*obj_mask[...,None] - dino_feats*object_mask.to(torch.float32)
            blurred_pix_loss_masked = sframe_pca_feats*obj_mask[...,None] - blur_dino_feats*object_mask.to(torch.float32)
            if mask_hands:
                pix_loss_masked = pix_loss_masked[shand_mask]
                blurred_pix_loss_masked = blurred_pix_loss_masked[shand_mask]
            dino_loss_masked = 1.0*pix_loss_masked.abs().mean() + 1.0*blurred_pix_loss_masked.abs().mean()
            loss['dino_loss_masked'] = dino_loss_masked
        
        else:
            loss['mask_loss'] = 0.
        
        if use_depth and self.use_depth:
           # This is ranking loss for monodepth (which is disparity)
            frame_depth = 1.0 / sframe_depth # convert disparity to depth
            N = 10000
            # erode the mask by like 10 pixels
            object_mask = dig_outputs['accumulation']>.9
            object_mask = object_mask & (~frame_depth.isnan())
            # commenting this out for now since it sometimes crashes with no valid pixels
            object_mask = kornia.morphology.erosion(
                object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device=self.device)
            ).squeeze().bool()
            if mask_hands:
                object_mask = object_mask & shand_mask
            valid_ids = torch.where(object_mask)
           
            if len(valid_ids[0]) > 0:
                rand_samples = torch.randint(
                    0, valid_ids[0].shape[0], (N,), device= self.device
                )
                rand_samples = (
                    valid_ids[0][rand_samples],
                    valid_ids[1][rand_samples],
                )
                rend_samples = dig_outputs["depth"][rand_samples]
                mono_samples = frame_depth[rand_samples]
                rank_loss = depth_ranking_loss(rend_samples, mono_samples)
                loss['depth_loss'] = self.rank_loss_mult * rank_loss
 
            else:
                loss['depth_loss'] = 0

        if use_rgb:
            rgb_loss =  0.05 *(dig_outputs['rgb']-srgb_frame)#.abs().mean()
            if mask_hands:
                rgb_loss = rgb_loss[shand_mask]
            rgb_loss = rgb_loss.abs().mean()
            loss['rgb_loss'] = rgb_loss
                
     
        if use_part_merging:
            connected_graph = self.pipeline.connected_graph
            if pose_deltas.shape[0] != len(connected_graph):
                part_pose_deltas = pose_deltas[:-1]
            else:
                part_pose_deltas = pose_deltas
            part_pose_deltas_t = part_pose_deltas[:,:3] 
            part_pose_deltas_r = quaternion_to_rotation_matrix(part_pose_deltas[:,3:])
            part_pose_deltas_transformation = translation_rotation_to_4x4_matrix(part_pose_deltas_t,part_pose_deltas_r)
            
            loss['part_merging_loss'] = 0

            indices = torch.nonzero(connected_graph, as_tuple=False)  # (N, 2)
            indices = indices[indices[:, 0] < indices[:, 1]] 
            i_indices, j_indices = indices[:, 0], indices[:, 1] 
            connected_points = self.pipeline.connected_points[i_indices, j_indices].clone().to(self.device)
            T_rp = translation_rotation_to_4x4_matrix(connected_points,torch.eye(3, device=self.device).expand(len(indices), 3, 3))  # Shape: (N, 4, 4)

            T_ri = torch.stack([self.relative_joint_poses[i].clone() for i in i_indices.tolist()])# Shape: (N, 4, 4)
            T_rj = torch.stack([self.relative_joint_poses[j].clone() for j in j_indices.tolist()])

            # Compute T_ip and T_jp
            T_ip = torch.linalg.inv(T_ri) @ T_rp  # (N, 4, 4)
            T_jp = torch.linalg.inv(T_rj) @ T_rp

            # Compute transformed poses
            T_ri_prime = T_ri @ part_pose_deltas_transformation[i_indices]  # (N, 4, 4)
            T_rj_prime = T_rj @ part_pose_deltas_transformation[j_indices]

            # Compute final transformations
            T_rp_prime1 = T_ri_prime @ T_ip  # (N, 4, 4)
            T_rp_prime2 = T_rj_prime @ T_jp

            # Compute distance metric in batch
            loss['part_merging_loss'] = transformation_matrix_distance(T_rp_prime1, T_rp_prime2, "geodesic", t_weight=10).sum()

    
        return loss,rendered_rgb


    def apply_to_model(self, pose_deltas, axis = None , vis_skeleton_tree = False):
        ###############################################################################################################
        self.reset_transforms()
        updated_joint_poses = {}
        delta_mat = pose2mat(pose_deltas).to(self.device)
        ###############################################################################################################
        for node,j_pose in self.relative_joint_poses.items():
            #we only consider rotation and translation for the root node
            # if node != self.root_label:
            #     delta_mat[node,:3,3] = torch.tensor([0,0,0],device=self.device)
            #j_pose : T_pc , delta_mat[node]: T_cc' 
            #T_pc' = T_pc @ T_cc'
            updated_joint_poses[node] = j_pose @ delta_mat[node] 
        ###############################################################################################################
        #run forward kinematics to update the joint positions
        world_poses = {}
        queue = deque([self.root_label])
        #make sure the root is updated first, update parent first then children
        ###############################################################################################################
        # time2 = time.time()
        while queue:
            current_node = queue.popleft()
            if current_node in self.tree:
                for child in self.tree[current_node]:
                    queue.append(child) 
            
            if current_node == self.root_label:
                world_poses[current_node] = updated_joint_poses[current_node]
            else:
                #T_wc = T_wp @ T_pc
                world_poses[current_node] = world_poses[self.parent_map[current_node]] @ updated_joint_poses[current_node]
        ###############################################################################################################
        #apply transformation at one time
        means = self.dig_model.gauss_params['means'].detach()
        rotmats = self.dig_model.gauss_params['quats'].detach()
        
        updated_curr_means = self.init_means.clone()
        updated_curr_rotmats = quat_to_rotmat(self.init_quats.clone())
        
        w_poses = self.w_poses.clone()
        
        # Extract nodes and poses from world_poses and move to the device
        for node,w_pose in world_poses.items():
            if node == self.root_label and  self.root_label ==len(self.input_group_masks):
                continue
            else:    
                w_poses[self.input_group_masks[node]] = w_pose
        
        local_transoform = self.local_gaussian_pose.clone()
       
        updated_transform = torch.matmul(w_poses,local_transoform)
        updated_curr_means = updated_transform[:,:3, 3]
        updated_curr_rotmats = updated_transform[:,:3,:3]

        means = updated_curr_means
        with torch.no_grad():
            rotmats = roma.rotmat_to_unitquat(updated_curr_rotmats) #x,y,z,w 
            rotmats = rotmats[:, [3, 0, 1, 2]]
        
        self.dig_model.gauss_params['means'] = means.float()
        self.dig_model.gauss_params['quats'] = rotmats.float()
        
        # visualize the model
        # we also need to update the axis direction

        if axis is not None:
            axis_w_poses = torch.tile(torch.eye(4),(self.num_joints,1,1)).to(self.device)
            for node,w_pose in world_poses.items():
                if node!=self.root_label:
                    axis_w_poses[node] = world_poses[self.parent_map[current_node]]
            
            #axis n_joints x 3 -> n_joints x 4
            local_axis = torch.cat([axis,torch.zeros((self.num_joints,1),device=self.device)],dim=1).unsqueeze(-1)
            updated_world_axis = (torch.bmm(axis_w_poses,local_axis).squeeze(-1))[:,:3]
            
            self.updated_joint_position = torch.stack([world_poses[i][:3,3] for i in sorted(world_poses.keys())]).to(self.device)
            self.pipeline.skeleton = self.build_skeleton(self.tree, self.updated_joint_position.clone().detach().cpu().numpy()) 

        if vis_skeleton_tree:
            self.updated_joint_position = torch.stack([world_poses[i][:3,3] for i in sorted(world_poses.keys())]).to(self.device)
            self.pipeline.skeleton = self.build_skeleton(self.tree, self.updated_joint_position.clone().detach().cpu().numpy()) 
      
    def build_skeleton(self,tree, joint_position):
        def find_children(tree, node_i):
            """
            Find all of node_i's children in the tree.
            """
            return tree.get(node_i, [])  
                
        skeleton= {}
        def traverse_and_add_line(parent , node , path_name):
            # Update the path name to include the current node
            current_path_name = f"{path_name}/{node}" if path_name else f"{node}"   
                
            line =  self.pipeline.viewer_control.viser_server.add_spline_catmull_rom(
                name = current_path_name,
                positions =np.array([joint_position[parent], joint_position[node]]) * VISER_NERFSTUDIO_SCALE_RATIO,
                tension = 0.5,
                line_width = 3.0,
                color=np.array([1.0, 0.0, 1.0]),
                segments = 100 )
            
            skeleton[node] = line
            
            # Recursively add controls for the children, updating the path name
            for child in find_children(tree, node):
                traverse_and_add_line(node , child, current_path_name)

        traverse_and_add_line(self.root_label, self.root_label, "")
        return skeleton   
    


    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params['means'] = self.init_means.clone()
            self.dig_model.gauss_params['quats'] = self.init_quats.clone()

    def load_init_camera_pose(self,video_idx):
        #load the camera 
        with torch.no_grad():
            video_init_cam = self.multi_video_data[video_idx]['init_c2w_matrix']
            h,w = video_init_cam['height'],video_init_cam['width']
            cx,cy = video_init_cam['cx'],video_init_cam['cy']
            fx,fy = video_init_cam['fx'],video_init_cam['fy']
            cam_pose = video_init_cam['camera_to_worlds']
            init_cam = Cameras(camera_to_worlds=cam_pose, fx = fx, fy = fy, cx = cx, cy = cy, width=w, height=h).to(self.device)
            self.init_c2w = init_cam
        return self.init_c2w

    def multi_video_data_init(self, video_lists, mask_lists=None, mask_hands = False, datatype ='real', n_seed=12, niter=400,):
        """
        Processes the video data for the given timestamps
        """
        self.data_type = datatype
        self.multi_video_data = []
        self.mask_lists = []
        self.original_mask_lists = []
        if mask_hands:
            self.hand_mask_lists = []
                
        if mask_lists is not None:
            self.original_mask_lists = mask_lists
            for masks in mask_lists:
                #turn 255 to 0-1 use clip
                masks = masks.clip(0,1)
                
                if masks.shape[1] > masks.shape[2]:
                    resized_frame_mask = resize(masks, (500, int(masks.shape[2]*(500/masks.shape[1])))) #(NF, H',W')
                else:
                    resized_frame_mask = resize(masks, (int(masks.shape[1]*(500/masks.shape[2])),500)) #(NF, H',W')
                
                self.mask_lists.append(resized_frame_mask)

        for i,video_path in enumerate(video_lists):
            assert video_path.exists() , "Video path does not exist"
            print(video_path)
            save_path = video_path.parent / video_path.stem / 'processed_video_list.pt' 
          
            
            if save_path.exists():
                processed_video = torch.load(save_path)
                self.multi_video_data.append(processed_video)

            else:
                #extract the camera information
                camera_id = int(video_lists[i].name.split('_')[-1].split('.')[0]) - 1

                #make the directory
                if mask_lists is not None:
                    self.first_frame_mask = self.mask_lists[i][0].to(self.device).to(torch.float32)
                motion_clip = cv2.VideoCapture(str(video_path.absolute()))
                frame_count = int(motion_clip.get(cv2.CAP_PROP_FRAME_COUNT))
                video_rgb = []
                video_depth = []
                video_dino_pca_feats = []
                processed_video = {}

          
                for frame_number in tqdm(range(frame_count)):
                    motion_clip.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    _,frame_rgb = motion_clip.read()
                 
                    if frame_number == 0:
                        #set the initial pose based on the first frame  
                        #update the initial camera to object transform
                        h,w = frame_rgb.shape[:2]
                        MATCH_RESOLUTION = 500
                        H = np.eye(4)
                        H[:3,:3] = vtf.SO3.from_x_radians(np.pi/4).as_matrix()
                        cam_pose = torch.from_numpy(H).float()[None,:3,:].to(self.device)

                        if datatype == 'real':
                            #we assume the input resolution is 1280x720 or 720x1280
                            if w > h :
                                init_cam = Cameras(camera_to_worlds=cam_pose.cpu(), fx =  1137.0,
                                                                                    fy = 1137.0,
                                                                                    cx = 1280/2,
                                                                                    cy = 720/2,
                                                                                    width = w,
                                                                                    height = h)
                            else:
                                init_cam = Cameras(camera_to_worlds=cam_pose.cpu(), fx =  1137.0,
                                                                                    fy =  1137.0,
                                                                                    cx =  720/2, 
                                                                                    cy =  1280/2,
                                                                                    width = w,
                                                                                    height = h)
                            # if the input resolution is not 1280x720 or 720x1280
                            if max(init_cam.width,init_cam.height) > 1280:
                                init_cam.rescale_output_resolution(init_cam, max(init_cam.width,init_cam.height)/1280)    
                        
                        elif datatype == 'synthetic':
                            # we assume the input resolution is 512*512
                            init_cam = Cameras(camera_to_worlds=cam_pose.cpu(), fx = 560.0,
                                                                                fy = 560.0,
                                                                                cx = 256.0,
                                                                                cy = 256.0,
                                                                                width = w,
                                                                                height = h)

                        init_cam.rescale_output_resolution(MATCH_RESOLUTION/max(init_cam.width,init_cam.height))
                        self.init_c2w = init_cam.to(self.device)
                        
                    
                    with torch.no_grad():
                        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB) #(H,W,3) 0-255
                        frame_rgb = ToTensor()(Image.fromarray(frame_rgb)).permute(1,2,0).to(self.device) #(H,W,3) 0.-1.
                        
                        frame_pca_feats = self.dino_loader.get_pca_feats(frame_rgb.permute(2,0,1).unsqueeze(0),keep_cuda=True).squeeze()
                        
                        depth = get_depth((frame_rgb*255).to(torch.uint8)) #(H,W)
                        frame_depth = resize(depth.unsqueeze(0), (self.init_c2w.height,self.init_c2w.width),antialias=True).squeeze().unsqueeze(-1) #(H',W')
                        resized_frame_rgb = resize(frame_rgb.permute(2,0,1), (self.init_c2w.height,self.init_c2w.width)).permute(1,2,0).contiguous() #(H',W',3)
                        
                    if frame_number == 0:
                        with torch.no_grad():
                            self.first_frame_rgb =  resized_frame_rgb #(H',W',3)
                            self.first_frame_pca_feats =  resize(frame_pca_feats.permute(2,0,1), (self.init_c2w.height,self.init_c2w.width)).permute(1,2,0).contiguous()#(H',W',D)
                            self.first_frame_depth = frame_depth

                      
                        if mask_hands:
                            hand_mask = get_hand_mask((self.first_frame_rgb* 255).to(torch.uint8))
                            hand_mask = (torch.nn.functional.max_pool2d(hand_mask[None, None], 3, padding=1, stride=1).squeeze()== 0.0)
                            self.first_frame_hand_mask = hand_mask.detach().clone()
                        else:
                            self.first_frame_hand_mask = None

                        initial_pose_image = self.initialize_camera_pose(render=True,niter=niter,n_seeds=n_seed, lr=0.06)
        
                        self.is_initialized = False
                        #save all the camera information
                        processed_video['init_c2w_matrix'] = {
                            'cx': self.init_c2w.cx.detach().cpu(),
                            'cy': self.init_c2w.cy.detach().cpu(),
                            'fx': self.init_c2w.fx.detach().cpu(),
                            'fy': self.init_c2w.fy.detach().cpu(),
                            'width': self.init_c2w.width.detach().cpu(),
                            'height': self.init_c2w.height.detach().cpu(),
                            'camera_to_worlds': self.init_c2w.camera_to_worlds.detach().cpu()
                        }
                        
                        save_path.parent.mkdir(parents=True,exist_ok=True) 
                        plt.imsave( Path(video_path.parent / video_path.stem / 'initial_pose.png' ), initial_pose_image)

                    video_rgb.append(resized_frame_rgb.detach().cpu())
                    video_depth.append(frame_depth.detach().cpu())
                    video_dino_pca_feats.append(frame_pca_feats.detach().cpu())
                    
                motion_clip.release()   
                save_path.parent.mkdir(parents=True,exist_ok=True) 
                
                processed_video['rgb'] = torch.stack(video_rgb)
                
                rgb_mean = processed_video['rgb'].mean(dim=(0,1,2),keepdim=True)
                processed_video['rgb_mean'] = rgb_mean
                    
                processed_video['depth'] = torch.stack(video_depth)
                processed_video['dino_pca_feats'] = torch.stack(video_dino_pca_feats)
                #save the processed video
                torch.save(processed_video,save_path)
                self.multi_video_data.append(processed_video)
                
                self.is_initialized = False
                self.first_frame_mask = None


            if mask_hands:
                save_path = video_path.parent / video_path.stem / 'processed_video_list.pt' 
                hand_mask_save_path = video_path.parent / video_path.stem / 'hand_mask_list.pt'
                
                if hand_mask_save_path.exists():
                    hand_mask_data = torch.load(hand_mask_save_path)
                    self.hand_mask_lists.append(hand_mask_data)

                else:
                    video_hand_mask = []
                    for i in tqdm(range(len(self.multi_video_data[-1]['rgb']))):
                        frame_rgb = self.multi_video_data[-1]['rgb'][i].to(self.device)
                        hand_mask = get_hand_mask((frame_rgb * 255).to(torch.uint8))
                        hand_mask = (torch.nn.functional.max_pool2d(hand_mask[None, None], 3, padding=1, stride=1).squeeze()== 0.0)
                        video_hand_mask.append(hand_mask.detach().cpu())
                    
                    #save the processed hand mask
                    video_hand_mask = torch.stack(video_hand_mask)
                    torch.save(video_hand_mask,hand_mask_save_path)
                    self.hand_mask_lists.append(video_hand_mask)
                        
    def downsample_video(self,scale :int = 1):
        if self.multi_video_data is not None:
            for v in range (len(self.multi_video_data)):
                video = self.multi_video_data[v]
                video['rgb'] =  video['rgb'][::scale]
                video['depth'] = video['depth'][::scale]
                video['dino_pca_feats'] = video['dino_pca_feats'][::scale]
                self.multi_video_data[v] = video
            
        if hasattr(self, 'mask_lists') and self.mask_lists is not None:
            for v in range(len(self.mask_lists)):
                self.mask_lists[v] = self.mask_lists[v][::scale]

        if hasattr(self, 'hand_mask_lists') and self.hand_mask_lists is not None:
            for v in range(len(self.hand_mask_lists)):
                self.hand_mask_lists[v] = self.hand_mask_lists[v][::scale]
        
    def cut_video(self, start_frame, end_frame):
        if self.multi_video_data is not None:
            for v in range (len(self.multi_video_data)):
                video = self.multi_video_data[v]
                video['rgb'] =  video['rgb'][start_frame:end_frame]
                video['depth'] = video['depth'][start_frame:end_frame]
                video['dino_pca_feats'] = video['dino_pca_feats'][start_frame:end_frame]
                self.multi_video_data[v] = video
            
        if hasattr(self, 'mask_lists') and self.mask_lists is not None:
            for v in range(len(self.mask_lists)):
                self.mask_lists[v] = self.mask_lists[v][start_frame:end_frame]

        if hasattr(self, 'hand_mask_lists') and self.hand_mask_lists is not None:
            for v in range(len(self.hand_mask_lists)):
                self.hand_mask_lists[v] = self.hand_mask_lists[v][start_frame:end_frame]