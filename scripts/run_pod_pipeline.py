import os
import torch
import numpy as np
from absl import app,flags
from ml_collections import config_flags
import json
from torch.utils.tensorboard import SummaryWriter

from pod.group_optimizer import Joint_RigidGroupOptimizer
from pod.utils.geo_utils import *
from pod.utils.train_utils import *
from pod.utils.ff_utils import *
from pod.pose_predictor import PosePredictor
from pod.pod_pipeline import *
from pod.configs.train_config import get_dataset_config  

import torch._dynamo
torch._dynamo.config.suppress_errors = True


FLAGS = flags.FLAGS
flags.DEFINE_bool("preload_camera", False, "use presaved camera")
flags.DEFINE_bool("pred_video_only", False, "load and predict video only")
flags.DEFINE_string("pred_model_path", '', "path to the pretrained model")
flags.DEFINE_string("run_name", '', "extra name for the run folder")
flags.DEFINE_bool("rsrd_video", False, "load and predict video only")
flags.DEFINE_string("rsrd_path", '', "path to the pretrained model")
flags.DEFINE_integer("video_idx",None, "select one video to train on")
flags.DEFINE_string("dataset_name", None, "Dataset name to load (e.g., toy2, jumping_lamp)")

def load_dataset_config():
    if not FLAGS.is_parsed():
        FLAGS([""])  # Parse with default values if not already parsed
    if FLAGS.dataset_name is None:
        raise ValueError("Dataset must be specified using --dataset=<dataset_name>")

    dataset_config = get_dataset_config(FLAGS.dataset_name)  # Get dataset-specific config
    return dataset_config


def save_flags_to_json(train_config,json_file_path):
    # Extract the flag values
    flags_dict = {}
    for flag_name in FLAGS:
        if flag_name in ['config']:
            continue
        flags_dict[flag_name] = FLAGS[flag_name].value
    
    flags_dict['config'] = train_config.to_dict()

    # Save to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(flags_dict, json_file, indent=4)


config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pod/configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "train_config.py:pod_pipeline_config"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

    
def main(_) -> None:
    # load config
    dataset_config = load_dataset_config()  
    gs_pipeline_config = FLAGS.config.gs_pipeline_config
    training_data_config = FLAGS.config.training_data_config
    pose_pred_train_config = FLAGS.config.pose_pred_train_config
    frame_matching_config = FLAGS.config.frame_matching_config
    multi_view_optim_config = FLAGS.config.multi_view_optim_config
    data_type = dataset_config.data_type
    
    if data_type == 'real':
        gs_pipeline_config.mask_hands = True
        pose_pred_train_config.use_patch_mask = True

    elif data_type == 'synthetic':
        gs_pipeline_config.mask_hands = False
        pose_pred_train_config.use_patch_mask = False
   
    # Load gs pipeline model
    v,gs_pipeline = gs_setup(dig_config_path = dataset_config.dig_config_path,
                        state_path = dataset_config.state_path,
                        rgb_state_path = dataset_config.rgb_state_path,
                        use_rgb_state=True,)
    
    video_folder = dataset_config.video_folder
    mask_folder = dataset_config.mask_folder
    video_list = list_mp4_files_in_directory(video_folder)
    video_list = sorted(video_list,key=lambda x: int(str(x).split('_')[-1][:-4]))

    # choose one long video
    selected_video_id = training_data_config.selected_video_id
    if FLAGS.video_idx is not None:
        selected_video_id = FLAGS.video_idx
  
    mask_lists = load_mask(mask_folder=mask_folder, video_list=video_list)
    init_cam = init_camera(data_type=data_type)
        
    if FLAGS.preload_camera:
        selected_cameras = torch.load('selected_cameras.pt')
    else:
        selected_cameras = select_camera_view(gs_pipeline)
        torch.save(selected_cameras, 'selected_cameras.pt')
    
    # initialize the joint_rigid_optimizer
    joint_rigid_optimizer = Joint_RigidGroupOptimizer(
            pipeline = gs_pipeline,
            init_c2w = init_cam,
            render_lock = v.train_lock,
            )
    
    joint_rigid_optimizer.multi_video_data_init(video_list,
                                                mask_lists,
                                                gs_pipeline_config.mask_hands,
                                                datatype = data_type,
                                                )

    joint_rigid_optimizer.reset_tree_structure()
    video_downsample_scale = training_data_config.video_downsample_scale
    joint_rigid_optimizer.downsample_video(scale = video_downsample_scale)

    if FLAGS.rsrd_video:
        gs_pipeline.eval()
        gs_pipeline.model.set_background(torch.ones(3))
        sequence_poses = torch.load(FLAGS.rsrd_path).detach()
        pred_partdeltas = sequence_poses.clone()[::training_data_config.sequence_downsample_scale]
        
        rsrd_video( joint_rigid_optimizer = joint_rigid_optimizer,
                    gs_pipeline = gs_pipeline,
                    output_folder_name = os.path.dirname(FLAGS.rsrd_path),
                    selected_cameras = selected_cameras,
                    pred_partdeltas = pred_partdeltas,
                    )
        return
    
    if FLAGS.pred_video_only:
        gs_pipeline.eval()
        gs_pipeline.model.set_background(torch.ones(3))
        pose_pred_model = PosePredictor(num_parts=joint_rigid_optimizer.num_joints,freeze_backbone=True).cuda()
        pose_pred_model.load_state_dict(torch.load(FLAGS.pred_model_path))
        curr_output_folder_name = os.path.dirname(FLAGS.pred_model_path)
        matching_frames,pred_results = predict_video( joint_rigid_optimizer = joint_rigid_optimizer,
                                            pose_pred_model = pose_pred_model,
                                            gs_pipeline = gs_pipeline,
                                            selected_video_id = selected_video_id,
                                            output_folder_name = curr_output_folder_name,
                                            frame_matching_config = frame_matching_config,
                                            selected_cameras = selected_cameras,
                                            )
        return
    
    # create folder
    if not os.path.exists("results"):
        os.mkdir("results")
    folder_name = 'results'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    day_time = datetime.now().strftime("%Y%m%d")
    hms_time = datetime.now().strftime("%H%M%S")
    folder_name = f'{folder_name}/{day_time}_{hms_time}_{FLAGS.dataset_name}_{FLAGS.run_name}'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # save config
    save_flags_to_json(FLAGS.config, os.path.join(folder_name, 'flags.json'))
    
    # load init training sequence
    if training_data_config.type == 'rsrd':
        if FLAGS.rsrd_path == '':
            raise ValueError("rsrd_path must be specified when training_data_config.type is 'rsrd'")
        else:
            training_pose_delta_path = FLAGS.rsrd_path
            print(f"Loading RS-RD poses from: {training_pose_delta_path}")
            if not os.path.exists(training_pose_delta_path):
                raise FileNotFoundError(f"RS-RD path {training_pose_delta_path} does not exist.")
            
        sequence_downsample_scale = training_data_config.sequence_downsample_scale
        sequence_poses = torch.load(training_pose_delta_path).detach() # n_frrames x n_parts x 7 (xyz_wxyz)
        sequence_poses[:, -1, :] = torch.tensor([0,0,0,1,0,0,0],dtype=torch.float32,device=sequence_poses.device) #do not consider the global transformation
        init_sequence_poses = sequence_poses[::sequence_downsample_scale]
    
    elif training_data_config.type == 'random':
        init_sequence_poses = generate_random_poses(200, joint_rigid_optimizer.num_joints, 0.02, np.pi/9)
        init_sequence_poses[:, -1, :] = torch.tensor([0,0,0,1,0,0,0],dtype=torch.float32,device=init_sequence_poses.device) #do not consider the global transformation
    
    elif training_data_config.type == 'static':
        init_sequence_poses = torch.zeros(200,joint_rigid_optimizer.num_joints,7)
        init_sequence_poses[:, :, :] = torch.tensor([0,0,0,1,0,0,0],dtype=torch.float32,device=init_sequence_poses.device)

    else:
        raise ValueError('training_data_config.type should be either rsrd or random')
    
    if FLAGS.config.video_cut == True:
        start_frame = FLAGS.config.time_range[0]
        end_frame = FLAGS.config.time_range[1]
        init_sequence_poses = init_sequence_poses[start_frame:end_frame]
        joint_rigid_optimizer.cut_video(start_frame,end_frame)
    
    writer = SummaryWriter(log_dir=folder_name)
    modelname = pose_pred_train_config.pose_pred_model_name 
    num_loop = FLAGS.config.num_loop
    start_loop = FLAGS.config.start_loop
    
    # iterative updaing loop
    skip_training = False
    for loop_idx in range(start_loop, num_loop):
        print('loop index: ',loop_idx)
        curr_output_folder_name = folder_name + f'/loop_{loop_idx}'
        if not os.path.exists(curr_output_folder_name):
            os.mkdir(curr_output_folder_name)
        print('curr_output_folder_name: ',curr_output_folder_name)

        if loop_idx == 0:
            #init loop
            curr_sequence_poses = init_sequence_poses
            pose_pred_model = PosePredictor(num_parts=joint_rigid_optimizer.num_joints,freeze_backbone=True).cuda()

            training_data =  generate_training_data( training_sequence_poses = curr_sequence_poses,
                                                    gs_pipeline= gs_pipeline,
                                                    joint_rigid_optimizer = joint_rigid_optimizer,
                                                    selected_video_id = selected_video_id,
                                                    output_folder_name = curr_output_folder_name,
                                                    training_data_config = training_data_config,
                                                    camera_pose_sampling_mode = 'spiral',
                                                    )
            
        elif loop_idx == start_loop and FLAGS.config.resume_path is not None:
            pose_pred_model = PosePredictor(num_parts=joint_rigid_optimizer.num_joints,freeze_backbone=True).cuda()
            pose_pred_model.load_state_dict(torch.load(FLAGS.config.resume_path))
            selected_video_camera_vec = torch.load(FLAGS.config.selected_video_camera_vec_path)
            if FLAGS.config.update_poses_path is not None:
                update_seq_poses = torch.load(FLAGS.config.update_poses_path)
            else:
                skip_training = True
            
            if not skip_training:
                curr_sequence_poses = update_seq_poses
        else:
            curr_sequence_poses = update_seq_poses
            selected_video_camera_vec = torch.load(folder_name + f'/loop_{loop_idx-1}/video_{selected_video_id}_camera_vec.pt')
            pose_pred_model = PosePredictor(num_parts=joint_rigid_optimizer.num_joints,freeze_backbone=True).cuda()
            if FLAGS.config.finetune:
                pose_pred_model.load_state_dict(torch.load(folder_name + f'/loop_{loop_idx-1}/pose_pred_model_{modelname}.pth'))
                selected_video_camera_vec = torch.load(folder_name + f'/loop_{loop_idx-1}/video_{selected_video_id}_camera_vec.pt')

        if loop_idx > 0:
            #update some config
            pose_pred_train_config.epochs = 150 
            # pose_pred_train_config.epochs = 200
            multi_view_optim_config.per_frame_step = max(50,multi_view_optim_config.per_frame_step-10)  ###### testing
            multi_view_optim_config.optimization_lr = max(5e-3, multi_view_optim_config.optimization_lr - 1e-3) ###### testing
            multi_view_optim_config.per_frame_root_step = max(10, multi_view_optim_config.per_frame_root_step-5) ###### testing

            if not skip_training:
                # generate training data    
                training_data =  generate_training_data( training_sequence_poses = curr_sequence_poses,
                                                        gs_pipeline= gs_pipeline,
                                                        joint_rigid_optimizer = joint_rigid_optimizer,
                                                        selected_video_id = selected_video_id,
                                                        output_folder_name = curr_output_folder_name,
                                                        training_data_config = training_data_config,
                                                        camera_pose_sampling_mode = training_data_config.camera_pose_sampling_mode,
                                                        predicted_camera_poses = selected_video_camera_vec,
                                                        data_type = data_type,
                                                        )

        if not skip_training:      
            # training
            pose_pred_model = pose_pred_train( pose_pred_model = pose_pred_model,
                                            data = training_data,
                                            modelname = modelname,
                                            output_folder_name = curr_output_folder_name,
                                            pose_pred_train_config = pose_pred_train_config,
                                            writer = writer,
                                            )

        # frame matching
        matching_frames_info , pred_results = predict_video( joint_rigid_optimizer = joint_rigid_optimizer,
                                                        pose_pred_model = pose_pred_model,
                                                        gs_pipeline = gs_pipeline,
                                                        selected_video_id = selected_video_id,
                                                        output_folder_name = curr_output_folder_name,
                                                        frame_matching_config = frame_matching_config,
                                                        selected_cameras = selected_cameras,
                                                        data_type = data_type,
                                                        )
        
            
        update_seq_poses = optim( joint_rigid_optimizer = joint_rigid_optimizer,
                                gs_pipeline = gs_pipeline,
                                all_pred_results = pred_results,
                                selected_video_id= selected_video_id,
                                output_folder_name = curr_output_folder_name,
                                multi_view_optim_config = multi_view_optim_config,
                                nms_matching_info  = matching_frames_info,
                                data_type = data_type,
                                writer = writer,
                                )
        skip_training = False


if __name__ == '__main__':
    app.run(main)