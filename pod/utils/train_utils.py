
import os
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import moviepy.editor as mpy

from pod.utils.geo_utils import *
import torch.nn.functional as F
import math

from pod.utils.utils import correct_t

import torch._dynamo
torch._dynamo.config.suppress_errors = True
import re
import random

from datetime import datetime

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import writer
import viser.transforms as vtf
from threading import Lock

random.seed(0)
np.random.seed(0) 
torch.manual_seed(0)


def extract_video_id(filename):
    # Use regex to find the last number in the string
    match = re.search(r'\d+(?!.*\d)', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No number found in the filename: {filename}")


def list_mp4_files_in_directory(directory_path):
    file_paths = []
    for file in os.listdir(directory_path):
        if file.endswith('.mp4'):
            file_paths.append(Path(os.path.join(directory_path, file)))
    return file_paths

def list_pt_files_in_directory(directory_path):
    file_paths = []
    for file in os.listdir(directory_path):
        if file.endswith('.pt'):
            file_paths.append(Path(os.path.join(directory_path, file)))
    return file_paths

def adjust_learning_rate(optimizer, epoch, min_lrs, max_lrs, warmup_epochs, total_epochs):
    
    if epoch < warmup_epochs:
        lrs = [max_lr * epoch / warmup_epochs for max_lr in max_lrs]
    else:
        lrs = [min_lr + (max_lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))) for min_lr, max_lr in zip(min_lrs, max_lrs)]
    
    if epoch > total_epochs:
        lrs = min_lrs
            
    for idx, param_group in enumerate(optimizer.param_groups):
        if "lr_scale" in param_group:
            param_group["lr"] = lrs[idx] * param_group["lr_scale"]
        else:
            param_group["lr"] = lrs[idx]
            
    return lrs

def adjust_lr(optimizer,initial_lr,final_lr,per_frame_step,current_step):
    lr = final_lr + (initial_lr - final_lr) * (1 - current_step / per_frame_step)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    

# def set_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

def set_learning_rate(optimizer, lrs):
    for i, lr in enumerate(lrs):
        optimizer.param_groups[i]['lr'] = lr
    

def set_decresing_learning_rate(optimizer, lrs , epoch, decay_rate = 0.1, decay_epoch = 100):
    lrs = [lr * (decay_rate ** (epoch // decay_epoch)) for lr in lrs]
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lrs[idx]
    return lrs

def multi_view_frame_selection(cur_embed, features, n_frames_per_video, n_selection = 2):
    '''
    cur-embed: (n_frames, embed_dim)
    features: (total_frames, embed_dim)
    '''
    start_time = time.time()
    norm_cur_embed = F.normalize(cur_embed, dim=-1)
    norm_features = F.normalize(features, dim=-1)
    
    similarity = norm_cur_embed@norm_features.T # (n_frames, total_frames)
    
    cum_idx = np.cumsum([0] + n_frames_per_video[:-1])
    similarity_per_video = []
    idx_per_video = []
    with torch.no_grad():
        for idx, n_frames in zip(cum_idx, n_frames_per_video):
            sim = similarity[:, idx:idx+n_frames]
            topk_v, topk_idx = torch.topk(sim, n_selection, dim=-1) # (n_frames, n_selection)
            similarity_per_video.append(topk_v)
            idx_per_video.append(topk_idx + idx)
    similarity_per_video = torch.stack(similarity_per_video, dim=0) # (num_video, n_frames, n_selection)
    similarity_per_video = similarity_per_video.permute(1, 0, 2).reshape(cur_embed.shape[0],-1) # (n_frames, num_video* n_selection)
    idx_per_video = torch.stack(idx_per_video, dim=0).permute(1, 0, 2).reshape(cur_embed.shape[0],-1)
    return similarity_per_video, idx_per_video

def multi_view_frame_selection_one_video(cur_embed, features, n_frames_per_video, n_selection = 2):
    '''
    cur-embed: (n_frames, embed_dim)
    features: (total_frames, embed_dim)
    '''
    norm_cur_embed = F.normalize(cur_embed, dim=-1)
    norm_features = F.normalize(features, dim=-1)
    
    similarity = norm_cur_embed@norm_features.T # (n_frames, total_frames)
     
    sampled_indices = torch.multinomial(similarity, num_samples=n_selection, replacement=False)
    return similarity.gather(1, sampled_indices), sampled_indices
    
    
    topk_v, topk_idx = torch.topk(similarity, n_selection, dim=-1) # (n_frames, n_selection)
    return topk_v, topk_idx

def sample_frame_pose(cur_pose, all_poses, root_label, n_selection = 2):
    #make poses rot 6d
    cur_pos_rot6d = quat_to_rot_6d(cur_pose[:, :, 3:].flatten(0,1)).reshape(cur_pose.shape[0], cur_pose.shape[1], -1)
    cur_pose_9d = torch.cat((cur_pose[:, :, :3],cur_pos_rot6d ), dim=-1)
    all_pos_rot6d = quat_to_rot_6d(all_poses[:, :, 3:].flatten(0,1)).reshape(all_poses.shape[0], all_poses.shape[1], -1)
    all_poses_9d = torch.cat((all_poses[:, :, :3],all_pos_rot6d ), dim=-1)
    
    #sample n_selection frames from all_poses for each frame in cur_pose
    all_joint_poses = torch.cat((all_poses_9d[:, :root_label], all_poses_9d[:, root_label+1:]), dim=1)
    all_joint_poses_norm = torch.norm(all_joint_poses, dim=(-2,-1))
    max_range = all_joint_poses_norm.max()*2
    
    root_pos = all_poses_9d[:, root_label]
    cur_pose_root = cur_pose_9d[:, root_label]
    root_distance = cur_pose_root[:, None].detach() - root_pos[None,:]
    root_distance = torch.norm(root_distance, dim=-1) # (n_frames, total_frames)
    #sample n_selection frames based on root distance
    sampled_indices = torch.multinomial(root_distance, num_samples=n_selection, replacement=False)
    # batch_size = cur_pose.shape[0]
    # sampled_indices = torch.randint(0, all_poses.shape[0], (batch_size, n_selection))
    sampled_poses = all_poses_9d[sampled_indices]
    distance = cur_pose_9d[:, None].detach() - sampled_poses
    #exclude root
    distance = torch.cat((distance[:, :, :root_label], distance[:, :, root_label+1:]), dim=2)
    distance = torch.norm(distance, dim=(-2,-1))
    return max_range - distance, sampled_indices

def sample_frame_time_latent(cur_embed, cur_idxes,  all_embeds, n_selection = 2, sample_num = 20):
    #make poses rot 6d
    time_distance = torch.abs(torch.Tensor(cur_idxes[:, None]) - torch.arange(all_embeds.shape[0])[None,:])
    sampled_indices = torch.multinomial(time_distance, num_samples=sample_num, replacement=False).to(cur_embed.device)
    sampled_embeds = all_embeds[sampled_indices]
    
    normalized_cur_embed = F.normalize(cur_embed, dim=-1)
    normalized_sampled_embeds = F.normalize(sampled_embeds, dim=-1)
    
    similarity = torch.einsum('ijk,ijk->ij', normalized_cur_embed[:,None].repeat(1,sample_num,1 ), normalized_sampled_embeds)
    topk_v, topk_idx = torch.topk(similarity, n_selection, dim=-1)
    return topk_v, sampled_indices.gather(1, topk_idx)

def ce_from_faraway_pose(cur_latent, cur_pose, all_latents, all_poses, root_label, n_selection = 2):
    #sample latents with different root T and enforce the latent to be close to the current pose latent if the joint poses are close
    #make poses rot 6d
    cur_pos_rot6d = quat_to_rot_6d(cur_pose[:, :, 3:].flatten(0,1)).reshape(cur_pose.shape[0], cur_pose.shape[1], -1)
    cur_pose_9d = torch.cat((cur_pose[:, :, :3],cur_pos_rot6d ), dim=-1)
    all_pos_rot6d = quat_to_rot_6d(all_poses[:, :, 3:].flatten(0,1)).reshape(all_poses.shape[0], all_poses.shape[1], -1)
    all_poses_9d = torch.cat((all_poses[:, :, :3],all_pos_rot6d ), dim=-1)
    
    root_pos = all_poses_9d[:, root_label]
    cur_pose_root = cur_pose_9d[:, root_label]
    root_distance = cur_pose_root[:, None].detach() - root_pos[None,:]
    root_distance = torch.norm(root_distance, dim=-1) # (n_frames, total_frames)
    #sample n_selection frames based on root distance
    sampled_indices = torch.multinomial(root_distance, num_samples=n_selection, replacement=True)

    sampled_poses = all_poses_9d[sampled_indices]
    distance = cur_pose_9d[:, None].detach() - sampled_poses
    #exclude root
    distance = torch.cat((distance[:, :, :root_label], distance[:, :, root_label+1:]), dim=2)
    distance = torch.norm(distance, dim=(-2,-1)) # (n_frames, n_selection)
    # distance_neg = torch.softmax(-distance, dim=-1)
    
    sampled_latents = all_latents[sampled_indices]
    cur_latent_expanded = cur_latent[:, None].repeat(1, n_selection,1)
    # latent_similarity = torch.softmax(torch.einsum('ijk,ijk->ij', cur_latent_expanded, sampled_latents), dim=-1) # (n_frames, n_selection)
    latent_similarity =torch.einsum('ijk,ijk->ij', cur_latent_expanded, sampled_latents) # (n_frames, n_selection)
    
    ce = F.cross_entropy(latent_similarity/ 0.07, distance, reduction='mean')
    return ce, sampled_indices
    # latent_similarity = torch.einsum('ijk,ijk->ij', cur_latent_expanded, sampled_latents) # (n_frames, n_selection)
    
    ce = -torch.sum(distance_neg * torch.log(latent_similarity + 1e-6), dim=-1)
    # ce = -torch.sum(latent_similarity * torch.log(distance_neg + 1e-6), dim=-1)
    return torch.mean(ce), sampled_indices

def ce_from_nearby_pose(cur_latent, cur_pose, all_latents,all_poses, root_label,  n_selection = 2):
    cur_pos_rot6d = quat_to_rot_6d(cur_pose[:, :, 3:].flatten(0,1)).reshape(cur_pose.shape[0], cur_pose.shape[1], -1)
    cur_pose_9d = torch.cat((cur_pose[:, :, :3],cur_pos_rot6d ), dim=-1)
    all_pos_rot6d = quat_to_rot_6d(all_poses[:, :, 3:].flatten(0,1)).reshape(all_poses.shape[0], all_poses.shape[1], -1)
    all_poses_9d = torch.cat((all_poses[:, :, :3],all_pos_rot6d ), dim=-1)
    
    similarity = cur_latent@all_latents.T # (n_frames, total_frames)
    sampled_indices = torch.multinomial(similarity, num_samples=n_selection, replacement=True) # (n_frames, n_selection)
    
    cur_root = cur_pose_9d[:, root_label]
    sampled_root = all_poses_9d[sampled_indices, root_label]
    root_distance = torch.norm(cur_root[:, None].detach() - sampled_root, dim=-1)
    # root_distance = torch.softmax(root_distance, dim=-1)
    
    sampled_latents = all_latents[sampled_indices]
    cur_latent_expanded = cur_latent[:, None].repeat(1, n_selection,1)
    # latent_similarity = torch.softmax(torch.einsum('ijk,ijk->ij', cur_latent_expanded, sampled_latents), dim=-1) # (n_frames, n_selection)
    latent_similarity =torch.einsum('ijk,ijk->ij', cur_latent_expanded, sampled_latents) # (n_frames, n_selection)
    
    ce = F.cross_entropy(latent_similarity/ 0.07, root_distance, reduction='mean')
    return ce, sampled_indices

    weighted_sim = (root_distance - torch.mean(root_distance, dim=-1)[:,None]) * latent_similarity # (n_frames, n_selection)
    return -torch.mean(weighted_sim), sampled_indices
    
    # ce = -torch.sum(root_distance * torch.log(latent_similarity + 1e-6), dim=-1)
    ce = -torch.sum(latent_similarity * torch.log(root_distance + 1e-6), dim=-1)
    
    return torch.mean(ce), sampled_indices
    
def save_videos(pod_model, 
                joint_rigid_optimizer, 
                no_root_transform, 
                init_pose_lists, 
                root_only, 
                n_frames_per_video, 
                save_folder, 
                video_list, 
                extra_name, 
                device,
                transformation_type = 'se3'):
    
    with torch.no_grad():
        for video_idx in tqdm(range(len(joint_rigid_optimizer.multi_video_data))):
            per_video_render_results = []   
            for frame_idx in range(len(joint_rigid_optimizer.multi_video_data[video_idx]['rgb'])):
                gt_poses = init_pose_lists[video_idx][frame_idx].to(device)
                rgb = joint_rigid_optimizer.multi_video_data[video_idx]['rgb'][frame_idx].to(device)
                
                embed_idx = sum(n_frames_per_video[:video_idx]) + frame_idx
                output,joints_rotation_origin, root_update = pod_model([embed_idx])
                
                trans = joints_rotation_origin[None].repeat(output.shape[0],1,1)
                rot6d = output[...,3:]
                quat = rot_6d_to_quat(rot6d)
                quat = quat / torch.norm(quat, dim=-1, keepdim=True)

                #correct rotation center
                if transformation_type == 'so3':
                    trans = correct_t(rotation_6d_to_matrix(rot6d),trans)
                elif transformation_type == 'se3':
                    trans =  output[..., :3]

                pose_deltas = torch.cat((trans, quat), dim=-1)
                if root_only:
                    pose_deltas = torch.zeros(1,joint_rigid_optimizer.num_joints, 7)
                    pose_deltas[:,:,3] = 1
                    pose_deltas = pose_deltas.to(device)
                

                if no_root_transform:
                    pose_deltas[:,joint_rigid_optimizer.root_label] = gt_poses[joint_rigid_optimizer.root_label]

                else:
                    root_t = root_update[:,:3]
                    root_q = rot_6d_to_quat( root_update[:,3:])
                    root_q = root_q / torch.norm(root_q, dim=-1, keepdim=True)
                    pose_deltas[:,joint_rigid_optimizer.root_label] = torch.cat((root_t,root_q),dim=-1)



                _,render_rgb = joint_rigid_optimizer.loss(video_idx=video_idx,
                                                        frame_idx=frame_idx,axis=None,
                                                        pose_deltas=pose_deltas[0],
                                                        use_rgb = False,
                                                        use_atap = False,
                                                        use_mask= False,
                                                        )

                per_video_render_results.append(((rgb.detach().cpu().numpy()*0.5+render_rgb*0.5)*255).astype(np.uint8))
                
            #save as an mp4
            fps=30
            output_video_dir = os.path.join(save_folder, f'{video_list[video_idx].stem}_decoder_output_{extra_name}.mp4')

            out_clip = mpy.ImageSequenceClip( per_video_render_results, fps=fps)
            out_clip.write_videofile( str(output_video_dir), fps=fps,)


def save_videos_incremental(pod_model, joint_rigid_optimizer, n_frames_per_video, save_folder, video_list, extra_name, device):
    with torch.no_grad():
        for video_idx in tqdm(range(len(joint_rigid_optimizer.multi_video_data))):
            per_video_render_results = []   
            for frame_idx in range(len(joint_rigid_optimizer.multi_video_data[video_idx]['rgb'])):
                rgb = joint_rigid_optimizer.multi_video_data[video_idx]['rgb'][frame_idx].to(device)
                
                embed_idx = sum(n_frames_per_video[:video_idx]) + frame_idx
                output,_, root_update = pod_model([embed_idx])
                
                trans = output[..., :3]
                quat = rot_6d_to_quat(output[...,3:])
                quat = quat / torch.norm(quat, dim=-1, keepdim=True)
                pose_deltas = torch.cat((trans, quat), dim=-1)
            
                root_t = root_update[:,:3]
                root_q = rot_6d_to_quat( root_update[:,3:])
                root_q = root_q / torch.norm(root_q, dim=-1, keepdim=True)
                pose_deltas[:,joint_rigid_optimizer.root_label] = torch.cat((root_t,root_q),dim=-1)

                _,render_rgb = joint_rigid_optimizer.loss(video_idx=video_idx,
                                                        frame_idx=frame_idx,axis=None,
                                                        pose_deltas=pose_deltas[0],
                                                        use_rgb = False,
                                                        use_atap = False,
                                                        use_mask= False,
                                                        )
                per_video_render_results.append(np.hstack(((rgb.detach().cpu().numpy()*255).astype(np.uint8),(render_rgb*255).astype(np.uint8))))
                
            #save as an mp4
            fps=30
            output_video_dir = os.path.join(save_folder, f'{video_list[video_idx].stem}_decoder_output_{extra_name}.mp4')

            out_clip = mpy.ImageSequenceClip( per_video_render_results, fps=fps)
            out_clip.write_videofile( str(output_video_dir), fps=fps,)
    
       
"""gs_points_save_function"""
def save_per_frame_gspoints(pod_model, joint_rigid_optimizer, no_root_transform, init_pose_lists,root_only,n_frames_per_video, save_folder, video_list, extra_name, device):
    with torch.no_grad():
        for video_idx in tqdm(range(len(joint_rigid_optimizer.multi_video_data))):
            per_video_render_results = []   
            output_gs_dir = os.path.join(save_folder,'points',f'{video_list[video_idx].stem}')
            print("save gspoints to ", output_gs_dir)
            if not os.path.exists(output_gs_dir):
                os.makedirs(output_gs_dir)

            for frame_idx in range(len(joint_rigid_optimizer.multi_video_data[video_idx]['rgb'])):
                gt_poses = init_pose_lists[video_idx][frame_idx].to(device)
                rgb = joint_rigid_optimizer.multi_video_data[video_idx]['rgb'][frame_idx].to(device)
                
                embed_idx = sum(n_frames_per_video[:video_idx]) + frame_idx
                output,joints_rotation_origin, root_update = pod_model([embed_idx])
                
                trans = joints_rotation_origin[None].repeat(output.shape[0],1,1)
                rot6d = output[...,3:]
                quat = rot_6d_to_quat(rot6d)
                quat = quat / torch.norm(quat, dim=-1, keepdim=True)

                #correct rotation center
                trans = correct_t(rotation_6d_to_matrix(rot6d),trans)

                pose_deltas = torch.cat((trans, quat), dim=-1)
                
                if root_only:
                    pose_deltas = torch.zeros(1,joint_rigid_optimizer.num_joints, 7)
                    pose_deltas[:,:,3] = 1
                    pose_deltas = pose_deltas.to(device)


                if no_root_transform:
                    pose_deltas[:,joint_rigid_optimizer.root_label] = gt_poses[joint_rigid_optimizer.root_label]

                else:
                    root_t = root_update[:,:3]
                    root_q =  rot_6d_to_quat(root_update[:,3:])
                    root_q = root_q / torch.norm(root_q, dim=-1, keepdim=True)
                    pose_deltas[:,joint_rigid_optimizer.root_label] = torch.cat((root_t,root_q),dim=-1)


                _,render_rgb = joint_rigid_optimizer.loss(video_idx=video_idx,
                                                        frame_idx=frame_idx,axis=None,
                                                        pose_deltas=pose_deltas[0],
                                                        use_rgb = False,
                                                        use_atap = False,
                                                        use_mask= False,
                                                        )

                per_video_render_results.append(((rgb.detach().cpu().numpy()*0.5+render_rgb*0.5)*255).astype(np.uint8))
                #save the gs points
                gaussian_points = joint_rigid_optimizer.dig_model.gauss_params['means'].detach().cpu().numpy()
                points = gaussian_points
                poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
                poses[:, :3, 3] = points
                poses = joint_rigid_optimizer.pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(torch.from_numpy(poses))
                gaussian_points = poses[:, :3, 3].numpy()
                np.save(os.path.join(output_gs_dir , f'gs_points_{frame_idx}.npy'),gaussian_points)


def gs_setup(dig_config_path, state_path, rgb_state_path = None, use_rgb_state = False, use_both = False):
    #load the gs pipeline
    config = Path(dig_config_path) 
    v_train_config,pipeline,_,_ = eval_setup(config)
    v = Viewer(ViewerConfig(default_composite_depth=False,num_rays_per_chunk=-1),config.parent,pipeline.datamanager.get_datapath(),pipeline,train_lock=Lock())
    v_train_config.logging.local_writer.enable = False
     # We need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly
    writer.setup_local_writer(v_train_config.logging, max_iter=v_train_config.max_num_iterations)
    pipeline.load_state_from_path(state_path_filename = state_path) 

    if use_both and use_rgb_state:
        pipeline.load_state_from_path(state_path_filename = state_path,blend_color = False)
        pipeline.load_state_from_path(state_path_filename = rgb_state_path,blend_color = True )
    elif not use_both and use_rgb_state:    
        if use_rgb_state and not rgb_state_path is None:
            pipeline.load_state_from_path(state_path_filename = rgb_state_path)


    #remove the previous joints and skeleton
    if pipeline.conjunction_joint is not None:
        for i in range(len(pipeline.conjunction_joint)):
            pipeline.conjunction_joint[i].remove()
    if pipeline.skeleton is not None:
        for i in range(len(pipeline.skeleton)):
            pipeline.skeleton[i].remove()
    return v,pipeline

def init_camera(init_c2w = None, data_type = "real"):
    #init default camera
    MATCH_RESOLUTION = 500
    H = np.eye(4)
    H[:3,:3] = vtf.SO3.from_x_radians(np.pi/4).as_matrix()
    cam_pose = torch.from_numpy(H).float()[None,:3,:]

    if data_type == "real": 
        if init_c2w is not None:
            cam_pose = init_c2w
        init_cam = Cameras(camera_to_worlds=cam_pose,
                        fx = 1137.0,
                        fy = 1137.0,
                        cx = 1280.0/2,
                        cy = 720./2,
                        width=1280,
                        height=720)
    elif data_type == "synthetic":
        if init_c2w is not None:
            cam_pose = init_c2w

        init_cam = Cameras(camera_to_worlds=cam_pose,
                        fx = 560.0,
                        fy = 560.0,
                        cx = 256.0,
                        cy = 256.0,
                        width = 512,
                        height = 512,
                            )
    init_cam.rescale_output_resolution(MATCH_RESOLUTION/max(init_cam.width,init_cam.height))
    return init_cam

def H_to_camera( H,
                data_type = "real" 
                ):
    
    if data_type == "real": 
        focal = 189.0
    elif data_type == "synthetic":
        focal = 245.0

    return Cameras(
        camera_to_worlds=torch.from_numpy(H[None,:3,:]).cuda().float(),
        fx=focal,
        fy=focal,
        cx=112.,
        cy=112.,
        width=224,
        height=224,
    )

def init_to_identity(joint_rigid_optimizer,pod_optimizer,pod_model,index,iter_num = 100):
    identity_pose_deltas = torch.zeros(pod_model.num_joints,7).to(joint_rigid_optimizer.device)
    identity_pose_deltas[:,3:] = torch.tensor([1,0,0,0],dtype=torch.float32,device=joint_rigid_optimizer.device)

    # Store original learning rates before training
    # Set the learning rate for training
    original_lrs = [param_group['lr'] for param_group in pod_optimizer.param_groups]
    reset_lr = 1e-1
    for param_group in pod_optimizer.param_groups:
        param_group['lr'] = reset_lr

    for i in tqdm(range(iter_num)):
        pod_optimizer.zero_grad()

        output,_,root_update = pod_model(index)
        rot6d = output[...,3:]
        quat = rot_6d_to_quat(rot6d)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        trans = output[...,:3]
        pose_deltas = torch.cat((trans, quat), dim=-1)
        root_t = root_update[:,:3] 
        root_q = rot_6d_to_quat(root_update[:,3:])
        root_q = root_q / torch.norm(root_q, dim=-1, keepdim=True)
        pred_root = torch.cat((root_t,root_q),dim=-1)
        pose_deltas[:,joint_rigid_optimizer.root_label] = pred_root
        pose_deltas = pose_deltas.squeeze(0)

        q_mse_loss = quaternion_mse_loss(pose_deltas[:,3:],identity_pose_deltas[:,3:])
        t_mse_loss = nn.MSELoss()(pose_deltas[:,:3],identity_pose_deltas[:,:3])
        total_loss = q_mse_loss + t_mse_loss
        total_loss.backward()
        pod_optimizer.step()
    
    print(pose_deltas)
    # Recover original learning rates after training
    for param_group, original_lr in zip(pod_optimizer.param_groups, original_lrs):
        param_group['lr'] = original_lr
    return 


def make_output_folder(FLAGS):
    init_train_config = FLAGS.config
    extra_name = 'atap_' if init_train_config.use_atap_loss else ''
    extra_name += 'rgb_' if init_train_config.use_rgb_loss else ''
    extra_name += 'mask_' if init_train_config.use_mask_loss else ''
    extra_name += 'simple_tree_' if init_train_config.use_simple_tree_structure else ''
    extra_name += 'hand_' if init_train_config.mask_hands else ''
    extra_name += 'attention_' if init_train_config.use_attention else ''
    
    day_time = datetime.now().strftime("%Y%m%d")
    hms_time = datetime.now().strftime("%H%M%S")
    output_dir = f"{FLAGS.save_path}/{init_train_config.model_type}/{day_time}/{init_train_config.model_name}/{extra_name}_{hms_time}"
    os.makedirs(output_dir, exist_ok=True)
    return init_train_config,extra_name,output_dir,day_time,hms_time

def load_mask(mask_folder,video_list):
    mask_lists = []
    
    for video_path in video_list:
        mask_path = os.path.join(mask_folder, str(video_path).split('/')[-1][:-4])
        mask_list = []
        files = os.listdir(mask_path)
        files = sorted(files,key=lambda x: int(str(x).split('_')[-1][:-4]))
        for file in files:
            if file.endswith('.npy'):
                mask_list.append(torch.from_numpy(np.load(os.path.join(mask_path,file))))
            else:
                raise ValueError(f"Mask file {file} in {mask_path} is not here")
        mask_lists.append(torch.stack(mask_list,dim=0))
    return mask_lists

