import os
import torch
import numpy as np
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
import moviepy.editor as mpy
import viser.transforms as vtf


from pod.utils.geo_utils import *
from pod.utils.train_utils import *
from pod.group_optimizer import Joint_RigidGroupOptimizer
from pod.utils.train_utils import  list_mp4_files_in_directory, adjust_lr

from absl import app,flags
from ml_collections import config_flags

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import random
random.seed(0)
np.random.seed(0) 
torch.manual_seed(0)

from pod.configs.train_config import get_dataset_config  
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_list("video_idx", [], "List of video indices to train on")

flags.DEFINE_string("dataset_name", None, "Dataset name to load (e.g., toy2, jumping_lamp)")
def load_dataset_config():
    if not FLAGS.is_parsed():
        FLAGS([""])  # Parse with default values if not already parsed
    if FLAGS.dataset_name is None:
        raise ValueError("Dataset must be specified using --dataset=<dataset_name>")

    dataset_config = get_dataset_config(FLAGS.dataset_name)  # Get dataset-specific config
    return dataset_config


config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pod/configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "train_config.py:init_config"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_) -> None: 

    # Load the init config
    init_train_config = FLAGS.config
    dataset_config = load_dataset_config()  
    init_train_config.update(dataset_config)

    # choose datatype based on the configuration
    if init_train_config.data_type == 'real':
        init_train_config.mask_hands = True
    elif init_train_config.data_type == 'synthetic':
        init_train_config.mask_hands = False

    #set name
    extra_name = init_train_config.init_type + '_'
    extra_name += 'rgb_' if init_train_config.use_rgb_loss else ''
    extra_name += '' if init_train_config.use_mask_loss else 'no_mask_'  
    extra_name += '' if init_train_config.use_depth_loss else 'no_depth_'
    extra_name += '' if init_train_config.use_part_merging_loss else 'no_part_merging_'
    extra_name += '' if init_train_config.mask_hands else 'no_mask_hands_'
    
    v,gs_pipeline = gs_setup(dig_config_path = init_train_config.dig_config_path,
                    state_path =  init_train_config.state_path,
                    rgb_state_path =  init_train_config.rgb_state_path,
                    use_rgb_state=True,)

    video_folder = init_train_config.video_folder
    mask_folder = init_train_config.mask_folder
    video_list = list_mp4_files_in_directory(init_train_config.video_folder)
    video_list = sorted(video_list,key=lambda x: int(str(x).split('_')[-1][:-4]))
    if len(FLAGS.video_idx) > 0:
        video_list = [video_list[int(idx)] for idx in FLAGS.video_idx]
    
    print(video_list)
    mask_lists = load_mask(mask_folder=mask_folder, video_list=video_list)
    init_cam = init_camera(data_type = init_train_config.data_type)
        
    joint_rigid_optimizer = Joint_RigidGroupOptimizer(
        pipeline = gs_pipeline,
        init_c2w = init_cam,
        render_lock = v.train_lock,
        )

    joint_rigid_optimizer.multi_video_data_init(video_list,
                                                mask_lists,
                                                init_train_config.mask_hands,
                                                init_train_config.data_type,
                                                FLAGS.config.n_seed,
                                                FLAGS.config.niter)
    
    #reset tree structure
    joint_rigid_optimizer.reset_tree_structure()
    
    #initialize the model(parameters and learning rate)
    if init_train_config.init_type == 'rsrd':
        initial_lr = 5e-3
 
    #add some visualization
    import plotly.express as px
    def plotly_render(frame):
        fig = px.imshow(frame)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),showlegend=False,yaxis_visible=False, yaxis_showticklabels=False,xaxis_visible=False, xaxis_showticklabels=False
        )
        return fig
    
    #initialization for visualization
    initial_rgb = joint_rigid_optimizer.multi_video_data[0]['rgb'][0].detach().cpu().numpy()    
    fig = plotly_render( initial_rgb)
    frame_vis = gs_pipeline.viewer_control.viser_server.add_gui_plotly(fig, 9/16)

    for video_idx in range(len(joint_rigid_optimizer.multi_video_data)):
        if init_train_config.init_type == 'rsrd':
            joint_rigid_optimizer.reset_pose_deltas()
            pose_optimizer = optim.Adam([joint_rigid_optimizer.pose_deltas], lr=initial_lr)
        
        idx_list = np.arange(len(joint_rigid_optimizer.multi_video_data[video_idx]['rgb']))
        per_video_init_pose = []
        per_video_render_results = []   
        for frame_idx in tqdm(idx_list):
            per_frame_step = init_train_config.per_frame_step
            
            for cur_step in range(per_frame_step):
                pose_optimizer.zero_grad()
                loss = {'dino_loss':0,  'depth_loss':0 , 'rgb_loss': 0, 'mask_loss': 0, 'part_merging_loss': 0}
                rgb = joint_rigid_optimizer.multi_video_data[video_idx]['rgb'][frame_idx]

                if init_train_config.init_type == 'rsrd':
                    axis = None
                    pose_deltas = joint_rigid_optimizer.pose_deltas
                    with torch.no_grad():
                        pose_deltas[:, 3:] = pose_deltas[:, 3:] /pose_deltas[:, 3:].norm(dim=1, keepdim=True)

                # # Calculate the loss
                loss_dict,render_rgb = joint_rigid_optimizer.loss(
                                                    video_idx = video_idx,
                                                    frame_idx = frame_idx,
                                                    axis= axis,
                                                    pose_deltas=pose_deltas,
                                                    use_rgb = init_train_config.use_rgb_loss,
                                                    use_mask = init_train_config.use_mask_loss,
                                                    mask_hands= init_train_config.mask_hands,
                                                    use_part_merging = init_train_config.use_part_merging_loss,
                                                    )
                                        
                    
                loss['dino_loss'] += 10 * loss_dict['dino_loss']

                if init_train_config.use_depth_loss:
                    loss['depth_loss'] += loss_dict['depth_loss']
                    total_loss = loss['dino_loss'] + loss['depth_loss'] 
                else:
                    total_loss = loss['dino_loss']
                
                if init_train_config.use_rgb_loss:
                    loss['rgb_loss'] += loss_dict['rgb_loss']
                    total_loss += loss['rgb_loss']
                    
                if init_train_config.use_mask_loss:
                    loss['mask_loss'] += loss_dict['mask_loss']
                    total_loss += 2.*loss['mask_loss']

                if init_train_config.use_part_merging_loss: 
                    loss['part_merging_loss'] += init_train_config.part_merging_weight*loss_dict['part_merging_loss'] 
                    total_loss += loss['part_merging_loss']
   
                total_loss.backward()
                pose_optimizer.step()

                if init_train_config.init_type == 'rsrd':
                    adjust_lr(optimizer = pose_optimizer,initial_lr = initial_lr,final_lr = initial_lr*0.1,per_frame_step = per_frame_step,current_step = cur_step)
              
                
            #real time visualization
            per_video_init_pose.append(pose_deltas.detach().cpu())   

            #concat images
            per_video_render_results.append(np.hstack((rgb.detach().cpu().numpy()*255, render_rgb.detach().cpu().numpy()*255)))

    
            fig = plotly_render(per_video_render_results[-1])
            frame_vis.figure = fig 
        
        #save init_pose_video
        #something wrong with the video saving
        fps=30    
        video_folder = Path(video_folder)
        output_video_dir = video_folder/ video_list[video_idx].stem / 'init'/ extra_name / 'init_output.mp4'
        if not os.path.exists(output_video_dir.parent):
            os.makedirs(output_video_dir.parent)
        out_clip = mpy.ImageSequenceClip( per_video_render_results, fps=fps)
        out_clip.write_videofile( str(output_video_dir), fps=fps,)
        
        #save init pose
        output_pose_dir = video_folder/ video_list[video_idx].stem / 'init'/ extra_name / 'init_pose.pt'
        if not os.path.exists(output_pose_dir.parent):
            os.makedirs(output_pose_dir.parent)
        per_video_init_pose = torch.stack(per_video_init_pose)
        torch.save(per_video_init_pose,output_pose_dir )    
   
if __name__ == "__main__":
    app.run(main)