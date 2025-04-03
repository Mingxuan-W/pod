from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from copy import deepcopy
import os
import numpy as np


def update_config(config, **kwargs):
    new_config = deepcopy(config)
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) or isinstance(config[key], ConfigDict):
                new_config[key] = update_config(config[key], **value)
            else:
                new_config[key] = value
        else:
            new_config[key] = value
    return ConfigDict(new_config)

def get_dataset_config(dataset_name):
    GLOBAL_PATH =os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define configurations for different datasets
    if dataset_name == "tractor":
        return ConfigDict({
            'dig_config_path': f'{GLOBAL_PATH}/outputs/tractor/dig/2025-03-30_095758/config.yml',
            'state_path': f"{GLOBAL_PATH}/state/tractor.pt",
            'rgb_state_path': f"{GLOBAL_PATH}/state/tractor_rgb.pt",
            'video_folder': f"{GLOBAL_PATH}/video/tractor",
            'mask_folder': f"{GLOBAL_PATH}/video/tractor_0/masks",
            'data_type': "real",  # or "synthetic"
        })
    elif dataset_name == "jumping_lamp":
        return ConfigDict({
            'dig_config_path': f'{GLOBAL_PATH}/outputs/jumping_lamp/dig/2024-10-07_102533/config.yml',
            'state_path': f"{GLOBAL_PATH}/state/jumping_lamp.pt",
            'rgb_state_path': f"{GLOBAL_PATH}/state/jumping_lamp_rgb.pt",
            'video_folder': f"{GLOBAL_PATH}/video/jumping_lamp",
            'mask_folder': f"{GLOBAL_PATH}/video/jumping_lamp/masks",
            'data_type': "synthetic",
        })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def get_config(config_string):
    init_config = dict(
        per_frame_step = 50,
        n_seed = 12,
        niter = 500,
        use_rgb_loss = False,
        use_mask_loss = True,
        use_depth_loss = True,
        use_part_merging_loss = True,
        part_merging_weight = 0.0003,
        init_type = "rsrd",
        mask_hands = True, # synthetic False real True
    )

    
    pod_pipeline_config = dict(
        start_loop = 0,
        num_loop = 5,
        finetune = True,
        resume_path = None,
        update_poses_path = None,
        selected_video_camera_vec_path = None,

        #ablation
        video_cut = False,
        time_range = [0,90],

        #gs_pipeline_config
        gs_pipeline_config = dict(
            use_atap_loss = False,
            use_rgb_loss = False,
            use_mask_loss = True,
            init_type = "rsrd",
            use_simple_tree_structure = True,
            simple_tree_type = "2", # 1 :old 2:n new
            mask_hands = True,
            use_atap_set = False,
            root_only = False,
        ),
        #training_data_config
        training_data_config = dict(
            type = "static",
            rand_rot_range = np.deg2rad(20),
            render_N_th = 600,  #####testing
            render_N_pi = 30,
            selected_video_id = 0, # we can set this in files
            video_downsample_scale = 1,
            sequence_downsample_scale = 1,
            
            random_camera = False, ##########!!!!!!!!!!!!!!
            aug_data = True,##########!!!!!!!!!!!!!!
            n_enhanced_camera = 10,
            camera_pose_sampling_mode = 'traj+spiral'
            
        ),
        #pose_pred_train_config
        pose_pred_train_config = dict(
            pose_pred_model_type = "dino",
            pose_pred_model_name = "config_acos_bigbatch_30reg",
            batchsize = 1500,
            # batchsize = 512,
            use_patch_mask = True,
            n_enhanced_feature = 3,
            epochs = 250,  #####testing
            # epochs = 1,  #####testing
            camera_loss_weight = 1.0,
            part_loss_weight = 1.0,
            # lr_init = 1e-4,
            lr_init = 1e-3,
            
            importance_sampling = False,
            img_aug = True, ###!!!!!!!!!!!!!!!! #let it to be default

            # new contrastive loss stuff
            contrastive_loss = False, ### testing
            contrastive_loss_weight = 0.5, 
            contrastive_loss_weight_decay = 0.85,

        ),
        #frame_matching_config
        frame_matching_config = dict(
            on = True,
            num_select = 4,
        ),
        #multi_view_optim_config
        multi_view_optim_config = dict(
            near_frame_range = 3,
            dino_w = 10 ,
            mask_w = 2,
            depth_w = 1,
            use_part_merging_loss = True,
            part_merging_w = 0.0003,
            per_frame_root_step = 15,
            per_frame_step = 30,  #####testing
            optimization_lr_root = 1e-2, 
            optimization_lr = 5e-3, 
            num_random_matching_frames = 1,
            batch_size = 20,
            
            smooth_start_epoch = 0,
            
            use_smooth_loss = True,
            smooth_loss_c = 0.01,
            smooth_loss_p = 0.1,
            smooth_loss_pos = 2.0,
            smooth_loss_rot = 1.0,
            smooth_loss_vel = 2.0,
            
            random_batch = True,
            use_velocity = True,
        ),
    )


    possible_structures = {
        "init_config": ConfigDict(
            init_config
        ),
        "pod_pipeline_config" : ConfigDict(
            pod_pipeline_config
        ),
    }

    return possible_structures[config_string]
