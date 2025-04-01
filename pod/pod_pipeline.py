import torch
import numpy as np
from tqdm import tqdm
from pod.utils.geo_utils import *

from pod.utils.train_utils import *
from pod.utils.ff_utils import *

import matplotlib.pyplot as plt
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
from viser.transforms import SO3, SE3
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import copy

def generate_training_data( training_sequence_poses,
                            gs_pipeline, 
                            joint_rigid_optimizer,
                            selected_video_id,
                            output_folder_name,
                            training_data_config,
                            camera_pose_sampling_mode = 'spiral',
                            predicted_camera_poses = None,
                            data_type = 'real',
                            ):
    """
    Generate synthetic training data for pose prediction model by rendering views from different camera positions.

    Inputs:
        - training_sequence_poses (torch.Tensor)
        - gs_pipeline
        - joint_rigid_optimizer
        - selected_video_id
        - output_folder_name
        - training_data_config
        - camera_pose_sampling_mode: 'spiral', 'traj', or 'spiral+traj'
        - predicted_camera_poses 
        - data_type : 'real' or 'synthetic'
    
    Returns:
        tuple: (views, object_masks, cam_poses, part_poses)
            - views
            - object_masks 
            - cam_poses 
            - part_poses

    """

    gs_pipeline.eval()
    gs_pipeline.model.set_background(torch.ones(3))
    joint_rigid_optimizer.reset_pose_deltas()
    obj_center = gs_pipeline.model.means.mean(dim=0)
    optimized_root_poses = training_sequence_poses[:,-1].clone()
    
    #reset all root pose
    training_sequence_poses[:,-1] = torch.tensor([0,0,0,1,0,0,0],device=training_sequence_poses.device)

    if 'spiral' in camera_pose_sampling_mode:
        #config
        init_c2w = joint_rigid_optimizer.load_init_camera_pose(selected_video_id).camera_to_worlds.clone()
        radius = init_c2w[0,:3,3].norm().item()
        render_radius_min = radius * 0.5
        render_radius_max = radius
        
        rand_rot_range = training_data_config.rand_rot_range
        render_N_th =  training_data_config.render_N_th
        render_N_pi = training_data_config.render_N_pi
        
        spiral_camera_poses = generate_spiral_poses(obj_center, 
                                            radius_min=render_radius_min,
                                            radius_max=render_radius_max, 
                                            n_theta=render_N_th, 
                                            n_phi=render_N_pi,
                                            rand_rot_range = rand_rot_range)
        # Convert to nerfstudio cameras
        spiral_cameras = [H_to_camera(pose.as_matrix(),data_type=data_type) for pose in spiral_camera_poses]
        spiral_views,spiral_cam_poses=[],[]
        for cam in tqdm(spiral_cameras):
            with torch.no_grad():
                gs_pipeline.model.set_background(torch.ones(3))
                view = gs_pipeline.model.get_outputs(cam)

            masked_rgb = view ['rgb'].clone()
            spiral_cam_poses.append(cam.camera_to_worlds[0].detach().cpu().numpy())

        # for each pose in camera_poses we're going to pick a keyframe to render at
        spiral_views,spiral_part_poses,spiral_object_masks= [],[],[]
        for p in tqdm(spiral_camera_poses):
            # choose a random keyframe
            key_id = np.random.randint(0, training_sequence_poses.shape[0])
            part_deltas =  training_sequence_poses[key_id].detach()
            with torch.no_grad():
                joint_rigid_optimizer.apply_to_model(part_deltas.to(joint_rigid_optimizer.device))
                view = gs_pipeline.model.get_outputs(H_to_camera(p.as_matrix(),data_type=data_type))
                masked_rgb = view ['rgb'].clone()
            spiral_views.append( masked_rgb.detach().cpu().numpy())
            object_mask = view ['accumulation']>.8
            spiral_object_masks.append(object_mask.detach().cpu().numpy())
            part_se3 = SE3(part_deltas.cpu().numpy()[:, [3, 4, 5, 6, 0, 1, 2]])
            spiral_part_poses.append(part_se3.as_matrix())

    if camera_pose_sampling_mode == 'spiral':
        views = spiral_views
        object_masks = spiral_object_masks
        cam_poses = spiral_cam_poses
        part_poses = spiral_part_poses

    elif 'traj' in camera_pose_sampling_mode:
        n_enhanced_camera = training_data_config.n_enhanced_camera
        traj_camera_poses = get_enhanced_optimized_camera_poses( predicted_camera_poses, optimized_root_poses, n_enhanced_camera//4)
        
        #unorder the traj camera poses
        traj_camera_poses = traj_camera_poses[torch.randperm(traj_camera_poses.shape[0])]
        
        traj_camera_poses_og = get_enhanced_optimized_camera_poses( predicted_camera_poses, optimized_root_poses, 0)
        traj_camera_poses_og_perturb = traj_camera_poses_og.clone().repeat_interleave(n_enhanced_camera//4, dim=0)
        traj_camera_poses_og_perturb = traj_camera_poses_og_perturb[torch.randperm(traj_camera_poses_og_perturb.shape[0])]
        traj_camera_poses_og = get_enhanced_optimized_camera_poses( predicted_camera_poses, optimized_root_poses, 0)
        traj_camera_poses_og_og = traj_camera_poses_og.clone().repeat_interleave(n_enhanced_camera - int(n_enhanced_camera//4)*2, dim=0)
        traj_camera_poses = torch.cat([traj_camera_poses,traj_camera_poses_og_perturb,traj_camera_poses_og_og ],dim=0)
        
   
        # Convert to nerfstudio cameras
        traj_cameras = [H_to_camera(pose.cpu().numpy(),data_type=data_type) for pose in traj_camera_poses]
        gs_pipeline.eval()
        gs_pipeline.model.set_background(torch.ones(3))
        traj_views,traj_cam_poses=[],[]
        for cam in tqdm(traj_cameras):
            with torch.no_grad():
                gs_pipeline.eval()
                gs_pipeline.model.set_background(torch.ones(3))
                view = gs_pipeline.model.get_outputs(cam)

            masked_rgb = view ['rgb'].clone()
            # traj_views.append( masked_rgb.detach().cpu().numpy())

            traj_cam_poses.append(cam.camera_to_worlds[0].detach().cpu().numpy())


        if n_enhanced_camera > 0:
            enhanced_sequence_poses = training_sequence_poses.repeat_interleave(n_enhanced_camera, dim=0)
        else:
            enhanced_sequence_poses = training_sequence_poses
        traj_views,traj_part_poses,traj_object_masks= [],[],[]
        
        for key_id,p in tqdm(enumerate(traj_camera_poses)):
            # choose a random keyframe
            part_deltas =  enhanced_sequence_poses[key_id].detach()
            with torch.no_grad():
                gs_pipeline.eval()
                gs_pipeline.model.set_background(torch.ones(3))
                joint_rigid_optimizer.apply_to_model(part_deltas.to(joint_rigid_optimizer.device))
                view = gs_pipeline.model.get_outputs(H_to_camera(p.cpu().numpy(),data_type=data_type))
                masked_rgb = view ['rgb'].clone()
            traj_views.append( masked_rgb.detach().cpu().numpy())
            object_mask = view ['accumulation']>.8
            traj_object_masks.append(object_mask.detach().cpu().numpy())
            part_se3 = SE3(part_deltas.cpu().numpy()[:, [3, 4, 5, 6, 0, 1, 2]])
            traj_part_poses.append(part_se3.as_matrix())
            
        clip = mpy.ImageSequenceClip([v*255 for v in traj_views], fps=30)
        clip.write_videofile(output_folder_name+"/dataset_config_traj.mp4")

        if camera_pose_sampling_mode == 'spiral+traj':
            #use part of spiral views + traj views
            views = spiral_views[::2] + traj_views
            object_masks = spiral_object_masks[::2] + traj_object_masks
            cam_poses = spiral_cam_poses[::2] + traj_cam_poses
            part_poses = spiral_part_poses[::2] + traj_part_poses
        else:
            views = traj_views
            object_masks = traj_object_masks
            cam_poses = traj_cam_poses
            part_poses = traj_part_poses

    return views,object_masks,cam_poses,part_poses

def pose_pred_train(pose_pred_model,
                    data,
                    modelname,
                    output_folder_name,
                    pose_pred_train_config,
                    writer
                    ):
    """
    Train a pose prediction model using rendered views and corresponding pose data.

    Inputs:
        - pose_pred_model
        - data (tuple): Training data (views, masks, camera poses, part poses)
        - modelname 
        - output_folder_name 
        - pose_pred_train_config
        - writer: TensorBoard writer for logging
    
    Returns:
        - pose_pred_model
    """

    #data
    views,object_masks,cam_poses,part_poses = data
    all_views = torch.from_numpy(np.stack(views)).cpu().permute(0,3,1,2)
    all_object_masks = torch.from_numpy(np.stack(object_masks)).cpu()
    all_poses = torch.from_numpy(np.stack(cam_poses)).cpu()
    all_parts = torch.from_numpy(np.stack(part_poses)).cpu().float()
    all_feats = []

    # config
    batchsize =  pose_pred_train_config.batchsize
    use_patch_mask =  pose_pred_train_config.use_patch_mask
    n_enhanced_feature = pose_pred_train_config.n_enhanced_feature
    epochs = pose_pred_train_config.epochs
    camera_loss_weight = pose_pred_train_config.camera_loss_weight
    part_loss_weight = pose_pred_train_config.part_loss_weight
    lr_init = pose_pred_train_config.lr_init
    
    with torch.no_grad():
        for i in range(0, len(views), batchsize):
            x = all_views[i:i+batchsize]
            x=  x.cuda()
            x = pose_pred_model.preprocess(x)
            all_feats.append(pose_pred_model.dino.get_intermediate_layers(x)[0].detach().cpu())
    all_feats = torch.cat(all_feats)
    
    all_patch_masked_feats =[]
    if use_patch_mask:
        print("Using patch mask")
        for _ in range(n_enhanced_feature):
            one_patch_masked_feats = []
            for i in range(0, len(views), batchsize):
                x = all_views[i:i+batchsize]
                x = add_patch_mask(x,all_object_masks[i:i+batchsize], min_patch_size = 30, max_patch_size = 50)  
                x= x.cuda()
                x = pose_pred_model.preprocess(x)
                one_patch_masked_feats.append(pose_pred_model.dino.get_intermediate_layers(x)[0].detach().cpu())
            one_patch_masked_feats = torch.cat(one_patch_masked_feats)
            all_patch_masked_feats.append(one_patch_masked_feats.unsqueeze(0))
        all_patch_masked_feats = torch.cat(all_patch_masked_feats)     
        
    all_auged_feats =[]
    if pose_pred_train_config.img_aug:
        print("Doing data aug")
        for i in range(0, len(views), batchsize):
            x = all_views[i:i+batchsize]
            x = img_aug(x, all_object_masks[i:i+batchsize]) 
            x= x.cuda()
            x = pose_pred_model.preprocess(x)
            all_auged_feats.append(pose_pred_model.dino.get_intermediate_layers(x)[0].detach().cpu())
        all_auged_feats = torch.cat(all_auged_feats)
        print("finish data aug")

    pose_pred_model.train()
    optimizer = torch.optim.Adam(pose_pred_model.parameters(), lr=lr_init)
    sched = ExponentialDecayScheduler(ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=epochs,
                    ramp="linear",
                    lr_pre_warmup=1e-6,
                    warmup_steps=20,
                    )).get_scheduler(optimizer, lr_init)

    losses = []
    tbar = tqdm(range(epochs), desc=f'Training ffmodel', unit='epoch')
    ids = np.random.permutation(len(all_views))
    train_ids = ids[:int(len(ids)*0.98)]
    val_ids = ids[int(len(ids)*0.98):]
    train_views = all_views[train_ids]
    val_views = all_views[val_ids]
    
    for epoch in tbar:
        all_pred_poses = []
        ids = np.random.permutation(len(train_views))
        for b in range(0, len(ids), batchsize):
            optimizer.zero_grad()
            b_ids = ids[b:b+batchsize]
            loaded_feats = copy.copy(all_feats[b_ids])
            if use_patch_mask and epoch % 2 == 0:
                patch_idx = np.random.choice(len(b_ids), len(b_ids)//2)
                loaded_feats[patch_idx] = all_patch_masked_feats[torch.randint(0, n_enhanced_feature-1, (1,)).item()][b_ids[patch_idx]]
            if pose_pred_train_config.img_aug:
                aug_idx = np.random.choice(len(b_ids), len(b_ids)//2)
                loaded_feats[aug_idx] = all_auged_feats[b_ids[aug_idx]]
            loaded_feats = loaded_feats.cuda()
 
            pred_obj2cam, pred_partdeltas = pose_pred_model.forward_from_dino(loaded_feats)# Bx4x4, BxNx4x4
            all_pred_poses.append(pred_obj2cam.detach().cpu().numpy())
            gt_poses = all_poses[b_ids].cuda()
            # invert them to predict world2cam 
            gt_poses = torch.cat([gt_poses, torch.tensor([0,0,0,1],dtype=torch.float32).cuda()[None,None].repeat(gt_poses.shape[0],1,1)],dim=1)#append 0001 to last row
            gt_poses = torch.linalg.inv(gt_poses)
            gt_parts = all_parts[b_ids].cuda()
            loss = camera_loss_weight * R_dist(pred_obj2cam[:,:3,:3], gt_poses[:,:3,:3]) 
            loss = loss + camera_loss_weight * (pred_obj2cam[:,:3,3] - gt_poses[:,:3,3]).norm(dim=-1).mean() # object translation
            loss = loss + part_loss_weight * R_dist(pred_partdeltas[...,:3,:3], gt_parts[...,:3,:3])
            loss = loss + part_loss_weight * (pred_partdeltas[...,:3,3] - gt_parts[...,:3,3]).norm(dim=-1).mean() # part translation
            loss.backward()
            #clip grads
            torch.nn.utils.clip_grad_norm_(pose_pred_model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item()/batchsize)
        sched.step()
        tbar.set_postfix(loss=loss.item()/batchsize)
        writer.add_scalar('FFmodel Loss/loss', loss.item()/batchsize, epoch)
        if epoch % 20 == 0:
            print("Saving model checkpoint")
            torch.save(pose_pred_model.state_dict(),output_folder_name+ f"/pose_pred_model_{modelname}.pth")
            
        if epoch % 1 == 0:
            #eval the model
            eval_loss = 0
            ids = np.random.permutation(len(val_views))
            with torch.no_grad():
                all_pred_poses = []
                for b in range(0, len(val_views), batchsize):
                    b_ids = ids[b:b+batchsize]
                    
                    loaded_feats = copy.copy(all_feats[b_ids])
                    if use_patch_mask:
                        patch_idx = np.random.choice(len(b_ids), len(b_ids)//2)
                        loaded_feats[patch_idx] = all_patch_masked_feats[torch.randint(0, n_enhanced_feature-1, (1,)).item()][b_ids[patch_idx]]
                    if pose_pred_train_config.img_aug:
                        aug_idx = np.random.choice(len(b_ids), len(b_ids)//2)
                        loaded_feats[aug_idx] = all_auged_feats[b_ids[aug_idx]]
                    loaded_feats = loaded_feats.cuda()
                    
                  
                    pred_obj2cam, pred_partdeltas = pose_pred_model.forward_from_dino(loaded_feats)
                    all_pred_poses.append(pred_obj2cam.detach().cpu().numpy())
                    
                    gt_poses = all_poses[b_ids].cuda()
                    # invert them to predict world2cam 
                    gt_poses = torch.cat([gt_poses, torch.tensor([0,0,0,1],dtype=torch.float32).cuda()[None,None].repeat(gt_poses.shape[0],1,1)],dim=1)#append 0001 to last row
                    gt_poses = torch.linalg.inv(gt_poses)
                    gt_parts = all_parts[b_ids].cuda()
                    eval_loss = camera_loss_weight * R_dist(pred_obj2cam[:,:3,:3], gt_poses[:,:3,:3]) 
                    eval_loss = eval_loss + camera_loss_weight * (pred_obj2cam[:,:3,3] - gt_poses[:,:3,3]).norm(dim=-1).mean() # object translation
                    
                    part_dist_rot =  R_dist(pred_partdeltas[...,:3,:3], gt_parts[...,:3,:3])
                    part_dist_trans = (pred_partdeltas[...,:3,3] - gt_parts[...,:3,3]).norm(dim=-1).mean() # part translation
                    eval_loss = eval_loss + part_loss_weight * part_dist_rot
                    eval_loss = eval_loss + part_loss_weight * part_dist_trans # part translation
                    
                writer.add_scalar('FFmodel Loss/Eval loss', eval_loss/len(val_views), epoch)
                
    # save the plot image
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.savefig(output_folder_name+ f"/pose_pred_loss.png")
    plt.close()

    return pose_pred_model

def predict_video( joint_rigid_optimizer,
                    pose_pred_model,
                    gs_pipeline,
                    selected_video_id ,
                    output_folder_name,
                    frame_matching_config,
                    selected_cameras,
                    data_type = 'real',
                   ):
    """
    Process a video to predict camera and joint poses for each frame.

    Inputs:
        - joint_rigid_optimizer
        - pose_pred_model
        - gs_pipeline
        - selected_video_id 
        - output_folder_name
        - frame_matching_config
        - selected_cameras 
        - data_type 'real' or 'synthetic'
    
    Returns:
        tuple: (matching_frames_info, all_pred_results)
            - matching_frames_info: NMS Frame matching information or None
            - all_pred_results
    """


    all_pred_joint_feature = []
    all_pred_camera_feature = []
    all_pred_camera_pose = []
    all_pred_pose_deltas = []
    all_pred_camera_vec = []
    for v in range(len(joint_rigid_optimizer.multi_video_data)):
        video_id = v
        per_video_pred_joint_feature = []
        per_video_pred_camera_feature = []
        per_video_pred_camera_poses = []
        per_video_pred_camera_vec = []
        per_video_pred_pose_deltas = []
        per_video_renderings = []
        pose_pred_model.eval()
        for i in tqdm(range(len(joint_rigid_optimizer.multi_video_data[video_id]['rgb']))):
            gs_pipeline.model.set_background(torch.ones(3))
            img = joint_rigid_optimizer.multi_video_data[video_id]['rgb'][i].clone().cuda()
            mask = joint_rigid_optimizer.mask_lists[video_id][i]    
            img[mask == 0] = torch.tensor([1.,1.,1.],device=img.device)
            img = square_image(img, square_type = 'pad+crop')
            # img = square_image(img, square_type = 'crop')
            img= img.permute(2,0,1).unsqueeze(0)
            with torch.inference_mode():
                pred_pose, pred_partdeltas = pose_pred_model(img)
                feature = pose_pred_model.per_frame_latent(img)
                per_video_pred_camera_feature.append(feature[:,0].detach().cpu())
                per_video_pred_joint_feature.append(feature[:,1].detach().cpu()) 

            part_7vec = SE3.from_matrix(pred_partdeltas.cpu().numpy().squeeze())
            wxyz_xyz = part_7vec.wxyz_xyz
            xyz_wxyz = wxyz_xyz[:, [4, 5, 6, 0, 1, 2, 3]]
            joint_rigid_optimizer.apply_to_model(torch.from_numpy( xyz_wxyz).cuda())
            per_video_pred_camera_vec.append(pred_pose.detach().cpu())
            pred_camera = H_to_camera(np.linalg.inv(pred_pose.cpu().numpy().squeeze()),data_type = data_type)
            per_video_pred_camera_poses.append( pred_camera)
            per_video_pred_pose_deltas.append(torch.from_numpy(xyz_wxyz).detach().cpu().unsqueeze(0))
            
            pred_rendering = gs_pipeline.model.get_outputs(pred_camera)
            
            #rendering from selected camera
            other_rendering = []
            for scam in selected_cameras:
                gs_pipeline.eval()
                gs_pipeline.model.set_background(torch.ones(3))
                other_render = gs_pipeline.model.get_outputs(H_to_camera(scam.as_matrix(),data_type = data_type))
                other_rendering.append(other_render['rgb'].clone().detach().cpu().numpy())
            
            out = np.zeros((img.shape[2],img.shape[3]*(2+len(selected_cameras)),3),dtype=np.float32)
            out[:,:img.shape[3],:] = img[0].permute(1,2,0).cpu().numpy()
            masked_rgb = pred_rendering['rgb'].clone()
            out[:,img.shape[3]:img.shape[3]*2,:] = masked_rgb.detach().cpu().numpy()
            for oidx, orender in enumerate(other_rendering):
                out[:,img.shape[3]*(2+oidx):img.shape[3]*(3+oidx),:] = orender
            
            per_video_renderings.append(out)

        clip = mpy.ImageSequenceClip([r*255 for r in per_video_renderings], fps=30)
        clip.write_videofile(output_folder_name + f"/processed_video_{video_id}_masking_long.mp4")
       

        all_pred_camera_pose.append(per_video_pred_camera_poses)

        per_video_pred_pose_deltas = torch.cat(per_video_pred_pose_deltas) 
        all_pred_pose_deltas.append(per_video_pred_pose_deltas)
        

        per_video_pred_camera_feature = torch.cat(per_video_pred_camera_feature)
        all_pred_camera_feature.append(per_video_pred_camera_feature)

        per_video_pred_joint_feature = torch.cat(per_video_pred_joint_feature)
        all_pred_joint_feature.append(per_video_pred_joint_feature) 
        
        per_video_pred_camera_vec = torch.cat(per_video_pred_camera_vec)
        all_pred_camera_vec.append(per_video_pred_camera_vec)

    #save selected video camera vector
    selected_pred_camera_vec = all_pred_camera_vec[selected_video_id].clone().detach().cpu()
    torch.save(selected_pred_camera_vec,output_folder_name + f'/video_{selected_video_id}_camera_vec.pt')

    # return seleceted video data
    selected_video_pred_pose_deltas = all_pred_pose_deltas[selected_video_id].clone().detach().cpu()
    torch.save(selected_video_pred_pose_deltas,output_folder_name + f'/video_{selected_video_id}_predict_poses.pt')
    joint_pose_heatmap_save_path = output_folder_name + f"/video_{selected_video_id}_joint_pose_heatmap.png"
    camera_pose_heatmap_save_path = output_folder_name + f"/video_{selected_video_id}_camera_pose_heatmap.png"
    
    selected_video_joint_poses = all_pred_pose_deltas[selected_video_id].clone()
    selected_video_joint_pos_rot6d = quat_to_rot_6d(selected_video_joint_poses[:, :, 3:].flatten(0,1)).reshape(selected_video_joint_poses.shape[0], selected_video_joint_poses.shape[1], -1)
    selected_video_joint_poses_9d = torch.cat((selected_video_joint_poses[:, :, :3],selected_video_joint_pos_rot6d), dim=-1)[:,:-1] # remove root pose
    joint_pose_distance,joint_pose_similarity = compute_frame_distance_matrix(selected_video_joint_poses_9d)
    visualize_distance_matrix(joint_pose_similarity , title='Joint Pose Similarity', save_path=joint_pose_heatmap_save_path)

    selected_video_camera_poses = all_pred_camera_vec[selected_video_id].clone()
    selected_video_camera_pos_rot6d = matrix_to_rotation_6d(selected_video_camera_poses[:, :3, :3])
    selected_video_camera_poses_9d = torch.cat((selected_video_camera_poses[:, :3, 3],selected_video_camera_pos_rot6d), dim=-1)
    camera_pose_distance,camera_pose_similarity = compute_frame_distance_matrix(selected_video_camera_poses_9d.unsqueeze(1))
    visualize_distance_matrix(camera_pose_similarity , title='Camera Pose Similarity', save_path=camera_pose_heatmap_save_path)

    if frame_matching_config.on:
        nms_matching_frames,nms_matching_frames_sim , nms_matching_frames_cam_dis  = nms_select_matching_frames(joint_pose_similarity, 
                                                                                                                camera_pose_distance, 
                                                                                    min_dist =  int(len(selected_video_camera_poses)/20), 
                                                                                    num_frames = int(len(selected_video_camera_poses)/4))
        
        return (nms_matching_frames,nms_matching_frames_sim,nms_matching_frames_cam_dis),(all_pred_joint_feature,
                                                                                        all_pred_camera_feature,
                                                                                        all_pred_camera_pose,
                                                                                        all_pred_camera_vec,
                                                                                        all_pred_pose_deltas)
        
    else:
        matching_frames_info = None
        return matching_frames_info,(all_pred_joint_feature,
                                all_pred_camera_feature,
                                all_pred_camera_pose,
                                all_pred_camera_vec,
                                all_pred_pose_deltas)

def optim(  joint_rigid_optimizer,
            gs_pipeline,
            all_pred_results,
            selected_video_id,
            output_folder_name,
            multi_view_optim_config,
            nms_matching_info = None,
            data_type = 'real',
            writer = None,
            ):

    """
    Optimize part poses for a video by refining initial predictions.

    Inputs:
        - joint_rigid_optimizer
        - gs_pipeline
        - all_pred_results (tuple): Prediction results from predict_video
        - selected_video_id 
        - output_folder_name 
        - multi_view_optim_config
        - nms_matching_info (tuple, optional): NMS Frame matching information
        - data_type: 'real' or 'synthetic'
        - writer: TensorBoard writer for logging
    
    Returns:
        - torch.Tensor: Optimized sequence poses 
    """

    gs_pipeline.eval()
    gs_pipeline.model.set_background(torch.ones(3))
    
    # data
    _,_,all_pred_camera_pose,_,all_pred_pose_deltas = all_pred_results
    if nms_matching_info is not None:
        nms_matching_frames,nms_matching_frames_sim,nms_matching_frames_cam_dis = nms_matching_info
        use_matching_frames = True
    else:
        use_matching_frames = False

    device = joint_rigid_optimizer.device   

    #config
    dino_w = multi_view_optim_config.dino_w
    mask_w = multi_view_optim_config.mask_w
    depth_w = multi_view_optim_config.depth_w
    use_part_merging_loss = multi_view_optim_config.use_part_merging_loss
    part_merging_w = multi_view_optim_config.part_merging_w
    per_frame_root_step = multi_view_optim_config.per_frame_root_step
    per_frame_step = multi_view_optim_config.per_frame_step
    optimization_lr = multi_view_optim_config.optimization_lr
    near_frame_range = multi_view_optim_config.near_frame_range
    num_random_matching_frames = multi_view_optim_config.num_random_matching_frames

    ##################################################################
    # optimize per-frame root pose
    ##################################################################
    root_optim_video = []
    all_view_root_pose_deltas = []
    for frame_id in tqdm(range(len(all_pred_pose_deltas[selected_video_id]))):
        selected_frame_id = frame_id
        all_frame_id = [selected_frame_id]
        curr_joint_pose_deltas = all_pred_pose_deltas[selected_video_id][selected_frame_id][:-1].detach().clone().to(device)
        per_view_root_pose_deltas = torch.stack([ torch.tensor([0,0,0,1,0,0,0],dtype=torch.float32,device=device) for _ in range(len(all_frame_id))])
        per_view_root_pose_deltas = torch.nn.Parameter(per_view_root_pose_deltas.clone())
        curr_pose_deltas_root_optimizer = torch.optim.Adam([per_view_root_pose_deltas], lr = optimization_lr*2 )

        for curr_step in range(per_frame_root_step):
            total_loss = 0
            curr_pose_deltas_root_optimizer.zero_grad()

            for i,id in enumerate(all_frame_id):
                gs_pipeline.eval()
                input_camera = all_pred_camera_pose[selected_video_id][id]
                input_pose_deltas = torch.cat([curr_joint_pose_deltas,per_view_root_pose_deltas[i].unsqueeze(0)],dim=0)
                joint_rigid_optimizer.apply_to_model(input_pose_deltas)  
                input_dig_outputs = gs_pipeline.model.get_outputs(input_camera)
                groundtruth = get_groundtruth_data(joint_rigid_optimizer,selected_video_id,id)
                loss,_= joint_rigid_optimizer.loss_calculation(
                                                    input_pose_deltas.cuda(),
                                                    input_dig_outputs, 
                                                    groundtruth, 
                                                    use_depth = True, 
                                                    use_rgb = False,
                                                    use_mask = True, 
                                                    mask_hands = True if data_type == 'real' else False,
                                                    use_part_merging = False,
                                                        )
                per_frame_loss = dino_w*loss['dino_loss'] + mask_w*loss['mask_loss'] + depth_w*loss['depth_loss']
                per_frame_vis = input_dig_outputs['rgb'].detach().cpu().numpy()*0.5 + groundtruth['rgb'].cpu().numpy()*0.5
                per_frame_vis = np.concatenate([groundtruth['rgb'].cpu().numpy(),input_dig_outputs['rgb'].detach().cpu().numpy(),per_frame_vis],axis=1)
                
                total_loss = per_frame_loss
                root_optim_video.append(per_frame_vis)
                
            total_loss.backward()
            curr_pose_deltas_root_optimizer.step()

        all_view_root_pose_deltas.append(input_pose_deltas[-1].detach().cpu())
    all_view_root_pose_deltas = torch.stack(all_view_root_pose_deltas).clone().cuda()
    clip = mpy.ImageSequenceClip([r*255 for r in root_optim_video], fps=30)
    clip.write_videofile(output_folder_name + f"/video_{selected_video_id}_root_optim.mp4")
    
    ##################################################################
    # optimize per-frame root + part pose
    ##################################################################
    updated_video = []
    all_view_root_pose_deltas = all_view_root_pose_deltas.clone().detach().cuda()
    all_view_root_pose_deltas = [torch.nn.Parameter(all_view_root_pose_deltas[id].clone().detach().cuda()) for id in range(len(all_view_root_pose_deltas))]
    root_optimizers =[ torch.optim.Adam([all_view_root_pose_deltas[id]], lr= optimization_lr) for id in range(len(all_view_root_pose_deltas))]

    all_joint_pose_deltas = [torch.nn.Parameter(all_pred_pose_deltas[selected_video_id][id,:-1].detach().clone().to(device)) for id in range(len(all_pred_pose_deltas[selected_video_id]))] 
    pose_deltas_optimizers = [torch.optim.Adam([all_joint_pose_deltas[id]], lr= optimization_lr ) for id in range(len(all_joint_pose_deltas))]
    
    part_sched = ExponentialDecayScheduler(ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=per_frame_step,
                    ramp="linear",
                    lr_pre_warmup=1e-6,
                    warmup_steps=0,
                    ))
    
    root_sched = ExponentialDecayScheduler(ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=per_frame_step,
                    ramp="linear",
                    lr_pre_warmup=1e-6,
                    warmup_steps=0,
                    ))
    
    if multi_view_optim_config.use_smooth_loss:
        part_scheds = [part_sched.get_scheduler(pose_deltas_optimizers[id], optimization_lr) for id in range(len(pose_deltas_optimizers))]
        root_scheds = [root_sched.get_scheduler(root_optimizers[id], optimization_lr) for id in range(len(root_optimizers))]
    
    
    allpbar = tqdm(range(per_frame_step), desc=f'Training All', unit='epoch')
    index_list = list(range(len(all_pred_pose_deltas[selected_video_id])))
    batch_size = multi_view_optim_config.batch_size
    
    
    for epoch_idx in allpbar:
        updated_video = [None]*len(index_list)
        
        if multi_view_optim_config.random_batch:
            index_list = np.random.permutation(index_list)
        if epoch_idx == per_frame_step-1:
            index_list = list(range(len(all_pred_pose_deltas[selected_video_id])))
            
        for batch_id in tqdm(range(0,len(index_list),batch_size)):
            batch_frame_ids = index_list[batch_id:batch_id+batch_size]
            opt_frame_ids = []
            
            total_loss = 0.0
            
            for frame_id in batch_frame_ids:
                pose_deltas_optimizers[frame_id].zero_grad()
                root_optimizers[frame_id].zero_grad()
                
                selected_frame_id = frame_id
                curr_joint_pose_deltas = all_joint_pose_deltas[selected_frame_id]
                
                all_frame_idx = [selected_frame_id] 
                if use_matching_frames:
                    nms_matching_frames_idx = nms_matching_frames[selected_frame_id].tolist()
                    nms_matching_frames_weight = min_max_normalize(nms_matching_frames_sim[selected_frame_id])*0.5+0.5 #0.5-1
                    nms_matching_frames_cam_dis_weight = min_max_normalize(nms_matching_frames_cam_dis[selected_frame_id])*0.5+0.5 #0.5-1
                    sampled_numbers = torch.multinomial(nms_matching_frames_cam_dis_weight, num_samples = num_random_matching_frames, replacement=True) 
                    all_frame_idx = all_frame_idx + [nms_matching_frames_idx[x] for x in sampled_numbers]
                    matching_frame_weights = [ nms_matching_frames_weight[x] for x in sampled_numbers ]
                
                opt_frame_ids += all_frame_idx
                
                for i,id in enumerate(all_frame_idx):
                    gs_pipeline.eval()
                    input_pose_deltas = torch.cat([curr_joint_pose_deltas,all_view_root_pose_deltas[id].unsqueeze(0)],dim=0)
                    joint_rigid_optimizer.apply_to_model(input_pose_deltas)  
                    input_camera = all_pred_camera_pose[selected_video_id][id]
                    input_dig_outputs = gs_pipeline.model.get_outputs(input_camera)
                    groundtruth = get_groundtruth_data(joint_rigid_optimizer,selected_video_id,id)
                    loss,_= joint_rigid_optimizer.loss_calculation(
                                                        input_pose_deltas,
                                                        input_dig_outputs, 
                                                        groundtruth, 
                                                        use_depth = True, 
                                                        use_rgb = False,
                                                        use_mask = True, 
                                                        mask_hands = True if data_type == 'real' else False,
                                                        use_part_merging = True,
                                                            )
                    #loss is None
                    if loss is None:
                        loss = {} 
                        loss['dino_loss'] = torch.tensor(0.0).cuda() if loss.get('dino_loss') is None else loss['dino_loss']
                        loss['mask_loss'] = torch.tensor(0.0).cuda() if loss.get('mask_loss') is None else loss['mask_loss']
                        loss['depth_loss'] = torch.tensor(0.0).cuda() if loss.get('depth_loss') is None else loss['depth_loss']
                        loss['part_merging_loss'] = torch.tensor(0.0).cuda() if loss.get('part_merging_loss') is None else loss['part_merging_loss']


                    per_frame_loss = dino_w*loss['dino_loss'] + mask_w*loss['mask_loss'] + depth_w*loss['depth_loss'] 
                    per_frame_vis = input_dig_outputs['rgb'].detach().cpu().numpy()*0.5 + groundtruth['rgb'].cpu().numpy()*0.5
                    
                    if use_part_merging_loss:
                        per_frame_loss = per_frame_loss  +  part_merging_w * loss['part_merging_loss'] 
                    
                    if id == selected_frame_id and i == 0:
                        per_frame_loss *= 5
                        total_loss += per_frame_loss
                        per_frame_vis = np.hstack([add_border(groundtruth['rgb'].cpu().numpy()),input_dig_outputs['rgb'].detach().cpu().numpy(),per_frame_vis])
                        render_rgb = per_frame_vis
                        updated_video[id] = per_frame_vis

                    elif id-selected_frame_id <-near_frame_range or id-selected_frame_id > near_frame_range:
                        total_loss +=  matching_frame_weights [i-1] * per_frame_loss 
                        per_frame_vis = np.hstack([groundtruth['rgb'].cpu().numpy(),input_dig_outputs['rgb'].detach().cpu().numpy(),per_frame_vis])
                        render_rgb = np.vstack([render_rgb,per_frame_vis])
                    
                    else:
                        per_frame_vis = np.hstack([groundtruth['rgb'].cpu().numpy(),input_dig_outputs['rgb'].detach().cpu().numpy(),per_frame_vis])
                        render_rgb = np.vstack([render_rgb,per_frame_vis*0.5])

            total_loss = total_loss / len(batch_frame_ids)
            writer.add_scalar('Frame Loss/dino_loss', total_loss.item(), epoch_idx)
                
            
            if multi_view_optim_config.use_smooth_loss and epoch_idx >= multi_view_optim_config.smooth_start_epoch:
                #smooth on part poses
                part_poses = pose2mat(torch.stack(all_joint_pose_deltas).reshape(-1,7))
                part_pos = part_poses[:,:3,3]
                
                if multi_view_optim_config.use_velocity:
                    pos_dis = part_pos[1:] - part_pos[:-1]
                    pos_vel = pos_dis[1:] - pos_dis[:-1]
                    pos_dist = multi_view_optim_config.smooth_loss_vel * torch.linalg.norm(pos_vel, dim=-1).mean() + torch.linalg.norm(pos_dis, dim=-1).mean()
                    rot_dis = torch.bmm(part_poses[1:,:3,:3], part_poses[:-1,:3,:3].permute(0,2,1))
                    rot_dist = multi_view_optim_config.smooth_loss_vel * R_dist(rot_dis[1:,:3,:3] , rot_dis[:-1,:3,:3]) + R_dist(part_poses[1:,:3,:3] , part_poses[:-1,:3,:3])
                else:
                    pos_dist = torch.linalg.norm(part_pos[1:] - part_pos[:-1], dim = -1).mean()
                    rot_dist = R_dist(part_poses[1:,:3,:3] , part_poses[:-1,:3,:3])
                
                pos_loss_p = pos_dist.mean()
                rot_loss_p =  rot_dist.mean()
                part_smooth_loss = multi_view_optim_config.smooth_loss_pos * pos_loss_p + multi_view_optim_config.smooth_loss_rot * rot_loss_p

                smooth_loss =  multi_view_optim_config.smooth_loss_p * part_smooth_loss
                total_loss = total_loss + smooth_loss
                

                writer.add_scalar('Frame Loss/part_smooth_loss_rot', rot_loss_p, epoch_idx)
                writer.add_scalar('Frame Loss/part_smooth_loss_pos', pos_loss_p, epoch_idx)
                writer.add_scalar('Frame Loss/part_smooth_loss', part_smooth_loss, epoch_idx)
                writer.add_scalar('Frame Loss/smooth_loss', smooth_loss, epoch_idx)
            
            total_loss.backward()
            for frame_id in batch_frame_ids:
                pose_deltas_optimizers[frame_id].step()
            for frame_id in opt_frame_ids:
                root_optimizers[frame_id].step()
                    
        allpbar.set_postfix(loss=total_loss.item(), refresh=True)
        if multi_view_optim_config.use_smooth_loss and epoch_idx > multi_view_optim_config.smooth_start_epoch:
            for part_s in part_scheds:
                part_s.step()
            for root_s in root_scheds:
                root_s.step()
        
        # clip = mpy.ImageSequenceClip([r*255 for r in updated_video], fps=30)
        # clip.write_videofile(output_folder_name + f"/updated_video_{selected_video_id}_optim_{epoch_idx}.mp4")  
    
    torch_new_seq_poses =torch.cat([torch.stack(all_joint_pose_deltas),torch.stack(all_view_root_pose_deltas)[:,None]],dim=1).detach().cpu()
    clip = mpy.ImageSequenceClip([r*255 for r in updated_video], fps=30)
    clip.write_videofile(output_folder_name + f"/updated_video_{selected_video_id}_optim.mp4")    
    torch.save(torch_new_seq_poses,output_folder_name + f'/video_{selected_video_id}_update_poses.pt')

    return torch_new_seq_poses   

def rsrd_video( joint_rigid_optimizer,
                gs_pipeline,
                output_folder_name,
                selected_cameras,
                pred_partdeltas,
                data_type = 'real',
            ):
    """
    Render a video using predicted part pose deltas from multiple camera views.

    Inputs:
        - joint_rigid_optimizer
        - gs_pipeline
        - output_folder_name 
        - selected_cameras
        - pred_partdeltas
        - data_type : 'real' or 'synthetic'
    
    Returns:
        None
    """

    partdeltas = pred_partdeltas.clone()
    partdeltas[:, -1, :] = torch.tensor([0,0,0,1,0,0,0],dtype=torch.float32,device=partdeltas.device) #do not consider the global transformation
    
    for v in range(len(joint_rigid_optimizer.multi_video_data)):
        video_id = v

        per_video_renderings = []
        for i in tqdm(range(len(joint_rigid_optimizer.multi_video_data[video_id]['rgb']))):
            gs_pipeline.model.set_background(torch.ones(3))
            img = joint_rigid_optimizer.multi_video_data[video_id]['rgb'][i].clone().cuda()
            mask = joint_rigid_optimizer.mask_lists[video_id][i]    
            img[mask == 0] = torch.tensor([1.,1.,1.],device=img.device)
            img = square_image(img, square_type = 'pad+crop')
            img= img.permute(2,0,1).unsqueeze(0)

            
            joint_rigid_optimizer.apply_to_model(pred_partdeltas[i])
            c2w = joint_rigid_optimizer.load_init_camera_pose(video_id)
            pred_rendering = gs_pipeline.model.get_outputs(c2w)
            
            #rendering from selected camera
            other_rendering = []
            joint_rigid_optimizer.apply_to_model(partdeltas[i])
            for scam in selected_cameras:
                gs_pipeline.model.set_background(torch.ones(3))
                other_render = gs_pipeline.model.get_outputs(H_to_camera(scam.as_matrix(),data_type=data_type))
                other_rendering.append(other_render['rgb'].clone().detach().cpu().numpy())
            
            out = np.zeros((img.shape[2],img.shape[3]*(2+len(selected_cameras)),3),dtype=np.float32)
            out[:,:img.shape[3],:] = img[0].permute(1,2,0).cpu().numpy()
            
            masked_rgb = pred_rendering['rgb'].clone()
            resized_img =square_image(masked_rgb, square_type = 'pad+crop')
            
            out[:,img.shape[3]:img.shape[3]*2,:] = resized_img.detach().cpu().numpy()
            
            for oidx, orender in enumerate(other_rendering):
                out[:,img.shape[3]*(2+oidx):img.shape[3]*(3+oidx),:] = orender

            per_video_renderings.append(out)

        clip = mpy.ImageSequenceClip([r*255 for r in per_video_renderings], fps=30)
        clip.write_videofile(output_folder_name + f"/rsrd_{video_id}.mp4")

def select_camera_view(pipeline):
    """
    Interactive utility to select camera views for multi-view rendering.

    Inputs:
        - pipeline: Rendering pipeline with viewer control
    
    Returns:
        - list: Selected camera poses as SE3 objects
    """
     
    server = pipeline.viewer_control.server
    selected_cameras = []
    
    #wait for the user to enter stop key
    while True:
        user_input = input("Enter 'y' to add selected cameras and 'n' to finish:")
        if user_input == 'y':
            pipeline.eval()
            pipeline.model.set_background(torch.ones(3))
            clients = server.get_clients()
            cam = clients[max(clients.keys())].camera
            campose = SE3.from_rotation_and_translation(SO3(cam.wxyz),cam.position/10) @ SE3.from_rotation(SO3.from_x_radians(np.pi))
            selected_cameras.append(campose)
        elif user_input == 'n':
            break
    
    return selected_cameras

