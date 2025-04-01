import torch
import numpy as np
from viser.transforms import SO3, SE3
from nerfstudio.cameras.cameras import Cameras
import cv2
import torchvision
import torchvision.transforms as transforms
from typing import Optional
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from pod.utils.geo_utils import *

def object_root_pose_to_c2w(init_object_root_pose, delta_object_root_pose, init_c2w):
    """
    Args:
        init_object_root_pose: (4 , 4)
        delta_object_root_pose: (4, 4) 
        init_c2w: (1, 3, 4)
    Returns:
        update_c2w: (1, 3, 4) update_T_wc
    """
    init_o2w = init_object_root_pose
    update_object_pose = torch.mm(init_object_root_pose, delta_object_root_pose)
    update_o2w = update_object_pose
    init_c2w = init_c2w.squeeze(0)
    init_c2w = torch.cat([init_c2w, torch.tensor([[0, 0, 0, 1]], device=init_c2w.device)], dim=0) 
    
    init_T_wo = init_o2w
    init_T_wc = init_c2w
    update_T_wo = update_o2w
    update_T_oc = torch.inverse(update_T_wo) @ init_T_wc
    update_T_wc = init_T_wo @ update_T_oc

    update_c2w =  update_T_wc[:3, :]
    update_c2w = update_c2w.unsqueeze(0)
    return update_c2w


def img_aug(images,
            object_masks
            ):
    """
    Apply image augmentation for images within the object masks.
    
    Args:
        images (torch.Tensor): Input RGB images of shape [n, 3, w, h].
        object_masks (torch.Tensor): Binary masks of shape [n, w, h], True for object regions.
    
    Returns:
        torch.Tensor: Augmented images.
    """
    n, c, w, h = images.shape
    augmented_images = images.clone()
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    augmented_images = color_jitter(augmented_images)
    # filter augmentation outside the mask
    augmented_images = augmented_images * object_masks.squeeze().unsqueeze(1)
    #sample some images to apply random color to the patch
    noisy_idx = np.random.choice(n, int(n*0.5), replace=False)
    for i in noisy_idx:
        # random mask small part of the image
        patch_w, patch_h = torch.randint(5, 20, (2,))
        num_patches = torch.randint(1, 3, (1,)).item()
        cx = torch.randint(0, w - patch_w, (num_patches,))
        cy = torch.randint(0, h - patch_h, (num_patches,))
        for j in range(num_patches):
            #assign random color to the patch
            color  = torch.rand(3)
            augmented_images[i, :, cx[j]:cx[j] + patch_w, cy[j]:cy[j] + patch_h] = color[:,None,None].repeat(1,patch_w,patch_h)
    return augmented_images


import torch

# need more update
def add_patch_mask( images, 
                    object_masks,
                    min_patch_size ,
                    max_patch_size,
                    patch_color=(1., 1., 1.)):
    """
    Add random rectangular patch masks within the masked regions of RGB images.
    
    Args:
        images (torch.Tensor): Input RGB images of shape [n, 3, w, h].
        patch_size (tuple): Size of the patch as (patch_width, patch_height).
        mask_color (tuple): RGB color representing the existing mask, e.g., (0, 0, 0).
        patch_color (tuple): RGB color to fill the new patch, e.g., (255, 0, 0).
    
    Returns:
        torch.Tensor: Batch of images with patch masks applied within the existing masked regions.
    """
    n, c, w, h = images.shape


    # Create a copy of the images to modify
    masked_images = images.clone()
    for i in range(n):
        patch_w, patch_h = torch.randint(min_patch_size, max_patch_size, (2,))
        num_patches = torch.randint(0, 3, (1,)).item()
        # Compute the binary mask where the existing masked regions are located
        mask = object_masks[i]  # Shape: [w, h], True for masked regions
        for _ in range(num_patches):
            # Find valid positions within the mask where the patch can fit
            valid_positions = mask.nonzero(as_tuple=False)  # Shape: [num_valid_positions, 2]
            if valid_positions.size(0) == 0:
                # If no valid position, skip
                continue

            # Randomly choose a starting point
            top_left_idx = torch.randint(0, valid_positions.size(0), (1,)).item()
            top_left = valid_positions[top_left_idx]  # Coordinates: (y, x)

            # Ensure the patch doesn't exceed the mask's bounds
            top = max(0, min(top_left[0], w - patch_w))
            left = max(0, min(top_left[1], h - patch_h))

            # Set the patch to the specified patch_color
            patch_color_tensor = torch.tensor(patch_color, device=images.device).view(3, 1, 1)  # Shape: [3, 1, 1]
            masked_images[i, :, top:top + patch_w, left:left + patch_h] = patch_color_tensor

    return masked_images

#predict camera pose matrix -> camera

def R_dist(R_pred, R_gt, eps = 1e-6):
    """
    distance between rotation matrices
    """
    R_pred = R_pred.reshape(-1, 3, 3)
    R_gt = R_gt.reshape(-1, 3, 3)
    diag = torch.eye(3,dtype=R_pred.dtype,device=R_pred.device)[None].repeat(R_pred.shape[0],1,1)
    trace = (diag * torch.bmm(R_pred, R_gt.permute(0,2,1))).sum(dim=(1,2))
    acos = torch.arccos(torch.clip((trace - 1) / 2, -1 + eps, 1 - eps))
    return acos.mean()


def get_vid_frame(
    cap: cv2.VideoCapture,
    timestamp: Optional[float] = None,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Get frame from video at timestamp (in seconds)."""
    if frame_idx is None:
        assert timestamp is not None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise ValueError("Video has unknown FPS.")
        frame_idx = min(
            int(timestamp * fps),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        )
    assert frame_idx is not None and frame_idx >= 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame at {timestamp} s, or frame {frame_idx}.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def find_closest_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> tuple[list[int], list[int]]:
    """
    Find the closest tensor pairs between two groups of tensors using cosine similarity.
    
    Args:
        tensor_a: First tensor of shape [n, 256]
        tensor_b: Second tensor of shape [m, 256]
        
    Returns:
        tuple containing:
            - list of indices from tensor_a showing closest match in tensor_b
            - list of indices from tensor_b showing closest match in tensor_a
    """
    # Normalize the tensors
    tensor_a_norm = F.normalize(tensor_a, p=2, dim=1)
    tensor_b_norm = F.normalize(tensor_b, p=2, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(tensor_a_norm, tensor_b_norm.t())
    
    # Find closest matches for tensor_a -> tensor_b
    closest_b = torch.argmax(similarity_matrix, dim=1).tolist()
    
    # Find closest matches for tensor_b -> tensor_a
    closest_a = torch.argmax(similarity_matrix, dim=0).tolist()
    
    return closest_b, closest_a

def find_k_closest_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor, k: int = 10) -> tuple[list[list[int]], list[list[int]], list[list[float]], list[list[float]]]:
    # Input validation
    k = min(k, tensor_a.shape[0], tensor_b.shape[0])
    
    # Normalize tensors
    tensor_a_norm = F.normalize(tensor_a, p=2, dim=1)
    tensor_b_norm = F.normalize(tensor_b, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(tensor_a_norm, tensor_b_norm.t())
    
    # Get top k matches for each tensor in a->b
    closest_b = torch.topk(similarity_matrix, k=k, dim=1)
    top_b_indices = closest_b.indices.tolist()
    top_b_scores = closest_b.values.tolist()
    
    # Get top k matches for each tensor in b->a
    similarity_matrix_t = similarity_matrix.t()
    closest_a = torch.topk(similarity_matrix_t, k=k, dim=1)
    top_a_indices = closest_a.indices.tolist()
    top_a_scores = closest_a.values.tolist()
    
    return top_b_indices, top_a_indices, top_b_scores, top_a_scores

def generate_spiral_poses(obj_center, 
                          radius_min, 
                          radius_max, 
                          n_theta, 
                          n_phi,
                          rand_rot_range, 
                          ):
    """
    Generate camera poses in a spiral pattern around an object center.
    
    Args:
        obj_center: torch.Tensor or np.ndarray of shape (3,) - Center point to orbit around
        radius: float - Distance from center to cameras
        n_theta: int - Number of points around the circle (horizontal)
        n_phi: int - Number of vertical levels
        
    Returns:
        list of SE3 transforms for camera positions
    """
    cameras = []
    
    # Convert obj_center to numpy if it's a tensor
    if hasattr(obj_center, 'detach'):
        obj_center = obj_center.detach().cpu().numpy()
    
    # Generate spiral points
    for j in range(n_phi):
        for i in range(n_theta):
            theta = (i / n_theta) * 2 * np.pi
            phi =  (j / (n_phi-1)) * np.pi/2
            
            # Calculate position on sphere
            radius = np.random.uniform(radius_min, radius_max)
            x = radius * np.cos(phi) * np.cos(theta)
            y = radius * np.cos(phi) * np.sin(theta)
            z = radius * np.sin(phi)
            
            # Camera position relative to object center
            position = np.array([x, y, z]) + obj_center
            
            # Create camera orientation - looking at object center
            look_dir = obj_center - position
            look_dir = look_dir / np.linalg.norm(look_dir)
            
            # Calculate up vector (approximately pointing upward)
            up = np.array([0, 0, 1])
            right = np.cross(look_dir, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, look_dir)
            
            # Create rotation matrix
            R = np.stack([right, up, -look_dir], axis=1)
            x_rot = np.random.uniform(-rand_rot_range, rand_rot_range)
            y_rot = np.random.uniform(-rand_rot_range, rand_rot_range)
            R = R @ SO3.from_x_radians(x_rot).as_matrix() @ SO3.from_y_radians(y_rot).as_matrix()
            
            # Create SE3 transform
            H = SE3.from_rotation_and_translation(SO3.from_matrix(R), position)
            cameras.append(H)
    
    return cameras

from scipy.spatial.transform import Rotation as R
def get_random_object_poses(n_frame,n_sample ,rot_x_range, rot_y_range, rot_z_range, translation_range):
    random_sample_poses = np.zeros((n_frame,n_sample ,4, 4))
    for i in range(n_frame):
        for j in range(n_sample):
            if j == 0:
                # Identity matrix
                random_sample_poses[i, j] = np.eye(4)
            else:
                # Random rotation with separate ranges
                angles = [
                    np.random.uniform(rot_x_range[0], rot_x_range[1]),  # X-axis
                    np.random.uniform(rot_y_range[0], rot_y_range[1]),  # Y-axis
                    np.random.uniform(rot_z_range[0], rot_z_range[1])   # Z-axis
                ]
                rot_mat = R.from_euler('xyz', angles, degrees=True).as_matrix()

                # Random translation
                translation = np.random.uniform(translation_range[0], translation_range[1], size=3)

                # Construct transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = translation

                random_sample_poses[i, j] = transform

    random_sample_poses = random_sample_poses.reshape(-1, 4, 4)
    return random_sample_poses


def get_enhanced_optimized_camera_poses(predicted_camera_poses, 
                                        optimized_root_poses , 
                                        n_enhanced_camera = 10,
                                        rot_x_range = (-10, 10),
                                        rot_y_range = (-10, 10), 
                                        rot_z_range = (-10, 10), 
                                        translation_range = (-0.1, 0.1)):
    """
    Convert a sequence of camera poses to a sequence of object poses.
    Args:
        predicted_camera_poses_camera_poses: (n_frames, 4, 4) tensor of camera poses (w2c) T_cw
        optimized_root_poses_joint_poses:  (n_frames, 7) tensor of joint poses
    Returns:
        optimized_camera_poses: (n_frames, 4, 4) tensor of camera poses
    """
    optimized_root_poses_T = pose2mat(optimized_root_poses)
    if n_enhanced_camera>0:
        random_object_poses_T = get_random_object_poses(len(predicted_camera_poses),n_enhanced_camera, rot_x_range, rot_y_range, rot_z_range, translation_range)
        enhanced_optimized_root_poses_T = optimized_root_poses_T.repeat_interleave(n_enhanced_camera, dim=0) @ random_object_poses_T 
        enhanced_optimized_camera_poses = predicted_camera_poses.repeat_interleave(n_enhanced_camera, dim=0) @ enhanced_optimized_root_poses_T.to(torch.float32) #T_co' -> T_cw'
    else:
        enhanced_optimized_camera_poses = predicted_camera_poses @ optimized_root_poses_T.to(torch.float32) #T_co' -> T_cw'
        
    enhanced_optimized_camera_poses = torch.linalg.inv(enhanced_optimized_camera_poses)
    return enhanced_optimized_camera_poses


def get_enhanced_object_poses( part_poses , 
                                        n_enhanced_poses= 10,
                                        rot_x_range = (-10, 10),
                                        rot_y_range = (-10, 10), 
                                        rot_z_range = (-10, 10), 
                                        translation_range = (-0.1, 0.1)):
    """
    Convert a sequence of camera poses to a sequence of object poses.
    Args:
        predicted_camera_poses_camera_poses: (n_frames, 4, 4) tensor of camera poses (w2c) T_cw
        optimized_root_poses_joint_poses:  (n_frames, 7) tensor of joint poses
    Returns:
        optimized_camera_poses: (n_frames, 4, 4) tensor of camera poses
    """
    T,N,C = part_poses.shape
    part_poses_T = pose2mat(part_poses.reshape(T*N,-1))
    
    random_object_poses_T = get_random_object_poses(len(part_poses_T),n_enhanced_poses, rot_x_range, rot_y_range, rot_z_range, translation_range)
    
    enhanced_part_poses_T = part_poses_T.repeat_interleave(n_enhanced_poses, dim=0) @ random_object_poses_T 
    
    enhanced_part_poses = mat2pose(enhanced_part_poses_T).reshape(-1,N,C)

    return enhanced_part_poses


def square_image(image: torch.Tensor, square_type: str, padding_value = 1 ) -> torch.Tensor:
    """
    Pad and resize input image to make it square.
    
    Args:
        image: Input image tensor of shape [H, W, m]
    Output:
        square_image: Padded and resized image tensor of shape [224, 224, m]
    """
    img = image.permute(2,0,1) #[m,H,W]
    h, w = img.shape[1:]
    
    # If already square, resize directly
    if h == w:
        img = torchvision.transforms.functional.resize(img, (224,224)).permute(1,2,0)
        return img
    
    # retectangle images
    if square_type == 'pad+crop':
        padding = (w - h) // 2
        if h > w:
            img_padding = (padding, padding,0,0) # (left, right, top, bottom)
            img = torch.nn.functional.pad(img, img_padding, value=padding_value, mode='constant')
        elif w > h:
            img_padding = (0, 0, padding, padding)  # (left, right, top, bottom)
            img = torch.nn.functional.pad(img, img_padding, value=padding_value, mode='constant')
        
        crop_length = max(h,w)//10
        img = img[:,crop_length: max(h,w)-crop_length,crop_length: max(h,w)-crop_length]
        img = torchvision.transforms.functional.resize(img, (224,224)).permute(1,2,0)   

    elif square_type == 'crop':
        img = img.unsqueeze(0) #[1,m,H,W]
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[2:]))
        img = torchvision.transforms.functional.resize(img, (224,224)).squeeze(0).permute(1,2,0) 

    else:
        raise ValueError(f"Invalid square_type: {square_type}")
    
    return img




def slerp_quaternions(q1, q2, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions
    Args:
        q1, q2: quaternions in (w,x,y,z) format
        t: interpolation parameter [0,1]
    """
    # Ensure unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate the dot product
    dot = np.dot(q1, q2)
    
    # If the dot product is negative, negate one of the quaternions
    # This ensures we take the shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    # Calculate interpolation coefficients
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def smooth_pose_sequence(poses, window_size=5):
    """
    Smooth a sequence of poses with multiple poses per frame
    Args:
        poses: numpy array of shape (n_frames, n_poses, 7) where each pose is [x,y,z,qw,qx,qy,qz]
        window_size: size of the smoothing window (odd number recommended)
    Returns:
        smoothed_poses: numpy array of same shape as input with smoothed poses
    """
    if not isinstance(poses, np.ndarray):
        poses = np.array(poses)
    
    n_frames, n_poses, n_dims = poses.shape
    assert n_dims == 7, f"Expected 7 dimensions per pose, got {n_dims}"
    
    smoothed_poses = np.zeros_like(poses)
    
    # Ensure window_size is odd
    window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
    half_window = window_size // 2
    
    # Smooth each pose sequence independently
    for p in range(n_poses):
        for i in range(n_frames):
            # Calculate window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(n_frames, i + half_window + 1)
            
            # Handle positions (simple moving average)
            positions = poses[start_idx:end_idx, p, :3]
            smoothed_poses[i, p, :3] = np.mean(positions, axis=0)
            
            # Handle orientations (quaternion averaging)
            center_quat = poses[i, p, 3:7]
            smoothed_quat = center_quat.copy()
            
            for j in range(start_idx, end_idx):
                if j != i:
                    t = 1.0 / (end_idx - start_idx)
                    smoothed_quat = slerp_quaternions(smoothed_quat, poses[j, p, 3:7], t)
            
            smoothed_poses[i, p, 3:7] = smoothed_quat / np.linalg.norm(smoothed_quat)
    
    return smoothed_poses


def compute_frame_distance_matrix(joint_poses):
    """
    Compute distance matrix between frames by flattening poses
    
    Args:
        joint_poses (torch.Tensor): Joint poses tensor of shape [T, M, D]
            T is number of frames/timesteps
            M is number of poses per frame
            D is feature dimension (9D in this case) translation + rotation(ro6d)
            
    Returns:
        torch.Tensor: Distance matrix of shape [T, T] containing distances between frames
    """
    T, M, D = joint_poses.shape
    
    # weight rotation and translation
    # joint_poses[:,:,:3] *= 10

    # Flatten poses: [T, M*D]
    flat_poses = joint_poses.reshape(T, M * D)
    
    # Reshape for broadcasting: [T, 1, M*D] and [1, T, M*D]
    poses_a = flat_poses.unsqueeze(1)  # [T, 1, M*D]
    poses_b = flat_poses.unsqueeze(0)  # [1, T, M*D]
    
    # Compute frame-to-frame distances directly
    frame_distances = torch.norm(poses_a - poses_b, dim=-1)  # [T, T]
    
    # Normalize by max_range
    flat_poses_norm = torch.norm(flat_poses, dim=-1)
    max_range = flat_poses_norm.max() * 2
        
    return frame_distances, max_range - frame_distances

def visualize_distance_matrix(distances, title="Frame Distance Matrix", figsize=(12, 10), 
                            cmap="YlOrRd", save_path=None):
    """
    Visualize the distance matrix using a heatmap
    
    Args:
        distances (torch.Tensor): Distance matrix [T*M, T*M] or [T, M, T, M]
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap for the heatmap
        save_path (str, optional): Path to save the figure
    """
    # Convert to numpy if tensor
    if isinstance(distances, torch.Tensor):
        distances = distances.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(distances, 
                cmap=cmap,
                xticklabels=True, 
                yticklabels=True,
                cbar_kws={'label': 'Distance'})
    
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    # plt.show()

def plot_one_frame_similarity(data, selected_frame_id, matching_frames, topk_matching_frames, title="Similarity Visualization"):
    """
    Plots frame similarity with highlighted frames.
    
    Parameters:
    - data: 1D array-like, similarity values to plot
    - selected_frame_id: int, the current frame to highlight
    - matching_frames: dict, maps frame indices to a list of matching frame indices
    - topk_matching_frames: dict, maps frame indices to a list of top-k matching frame indices
    - title: str, title for the plot
    """
    highlight_frames_1 = matching_frames[selected_frame_id]  # First group (Blue)
    highlight_frames_2 = topk_matching_frames[selected_frame_id]  # Second group (Orange)
    current_frame = selected_frame_id  # Current frame (Green)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the 1D data as a line graph
    ax.plot(data, color="red", linewidth=2, label="Similarity")
    
    # Highlight first group of frames (Blue)
    for frame in highlight_frames_1:
        ax.axvspan(frame - 0.5, frame + 0.5, color="blue", alpha=0.3, label="Matching Frames" if frame == highlight_frames_1[0] else "")
    
    # Highlight second group of frames (Orange)
    for frame in highlight_frames_2:
        ax.scatter(frame, data[frame], color="yellow", s=100, edgecolors="black")
    
    # Highlight the current frame with a vertical green line
    ax.axvline(current_frame, color="green", linestyle="--", linewidth=2, label="Current Frame")
    
    # Mark the current frame with a dot
    ax.scatter(current_frame, data[current_frame], color="green", s=100, edgecolors="black", label="Current Frame Point")
    
    # Customize the plot
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Similarity")
    ax.set_title(title)
    ax.legend()
    
    # Show the plot
    plt.show()

def topk_select_matching_frames(frame_joint_similarities, frame_camera_distances, top_k=10, num_select=4):
    """
    Select matching frames using a two-step process:
    1. Select top_k frames based on joint pose similarity
    2. From these top_k frames, select the num_select frames with largest camera distances
    
    Args:
        frame_joint_similarities: [T, T] tensor of joint pose similarities
        frame_camera_distances: [T, T] tensor of camera pose distances
        top_k: number of top similar poses to consider
        num_select: number of frames to select based on camera distance
        
    Returns:
        torch.Tensor: indices of selected frames [T, num_select]
    """
    T = frame_joint_similarities.shape[0]
    
    # Ensure valid parameters
    top_k = min(top_k, T-1)  # -1 to exclude self
    num_select = min(num_select, top_k)
    
    # Step 1: Get top_k similar frames based on joint similarity
    _, top_k_indices = torch.topk(frame_joint_similarities, k=top_k + 1, dim=1)
    
    # Remove self-reference (assuming it's always the most similar)
    top_k_indices = top_k_indices[:, 1:top_k + 1]
    
    # Step 2: Get camera distances for the top_k frames
    selected_camera_distances = torch.gather(
        frame_camera_distances, 
        1, 
        top_k_indices
    )
    
    # Select frames with maximum camera distances
    _, min_distance_indices = torch.topk(selected_camera_distances, k=num_select, dim=1)
    final_indices = torch.gather(top_k_indices, 1, min_distance_indices)
    
    return final_indices, top_k_indices

def nms_select_matching_frames(
    frame_joint_similarities: torch.Tensor,  # Similarity scores for frames
    camera_distances: torch.Tensor,  # Camera similarity scores
    min_dist: int = 5,  # Minimum frame distance constraint
    num_frames: int = 10  # Number of matching frames per frame
):
    num_total_frames = len(frame_joint_similarities)
    matching_frames = []
    matching_frames_similarity = []
    matching_frames_camera_distances = []
    num_frames+=1

    for i in range(num_total_frames):
        curr_frame_indices = torch.argsort(frame_joint_similarities[i], descending=True)
        selected_frame_idx = []
        selected_frame_joint_similarity = []
        selected_frame_camera_distance = []

        for idx in curr_frame_indices:
            if len(selected_frame_idx) >= num_frames:
                break
            if all(abs(idx - s) >= min_dist for s in selected_frame_idx):
                selected_frame_idx.append(idx.item())
                selected_frame_joint_similarity.append(frame_joint_similarities[i, idx].item())
                selected_frame_camera_distance.append(camera_distances[i, idx].item())        

        # If not enough, select the rest
        if len(selected_frame_idx) < num_frames:
            for idx in curr_frame_indices:
                if len(selected_frame_idx) >= num_frames:
                    break
                if idx.item() not in selected_frame_idx:
                    selected_frame_idx.append(idx.item())
                    selected_frame_joint_similarity.append(frame_joint_similarities[i, idx].item())    
                    selected_frame_camera_distance.append(camera_distances[i, idx].item())
        
        matching_frames.append(torch.tensor(selected_frame_idx))
        matching_frames_similarity.append(torch.tensor(selected_frame_joint_similarity))
        matching_frames_camera_distances.append(torch.tensor(selected_frame_camera_distance))
    
    matching_frames = torch.stack(matching_frames)[:,1:]
    matching_frames_similarity = torch.stack(matching_frames_similarity)[:,1:]
    matching_frames_camera_distances = torch.stack(matching_frames_camera_distances)[:,1:]
    
    return matching_frames, matching_frames_similarity, matching_frames_camera_distances

def get_groundtruth_data(joint_rigid_optimizer,video_id,frame_id,square_type ='pad+crop'):
    rgb = joint_rigid_optimizer.multi_video_data[video_id]['rgb'][frame_id].clone()
    rgb[ joint_rigid_optimizer.mask_lists[video_id][frame_id] == 0] = torch.tensor([1.,1.,1.],device=rgb.device)

    rgb = square_image(rgb,square_type = square_type)

    dino = joint_rigid_optimizer.multi_video_data[video_id]['dino_pca_feats'][frame_id]
    dino = square_image(dino,square_type = square_type)

    depth = joint_rigid_optimizer.multi_video_data[video_id]['depth'][frame_id]
    depth = square_image(depth,square_type = square_type)

    object_mask = joint_rigid_optimizer.mask_lists[video_id][frame_id]
    object_mask = square_image(object_mask.unsqueeze(-1),square_type = square_type,padding_value=0).squeeze(-1)
    
    if hasattr(joint_rigid_optimizer, 'hand_mask_lists') and joint_rigid_optimizer.hand_mask_lists is not None:
        hand_mask = joint_rigid_optimizer.hand_mask_lists[video_id][frame_id]
        hand_mask = square_image(hand_mask.unsqueeze(-1),square_type = square_type,padding_value=1).squeeze(-1)
    else:
        hand_mask = None

    groundtruth = {
        'rgb':rgb,
        'dino_pca_feats':dino,
        'depth':depth,
        'object_mask':object_mask,
        'hand_mask':hand_mask
    }
    return groundtruth

def add_border(image, border_width=5, border_color=(1., 0, 0)):
    """
    Add a colored border to an image represented as a NumPy array.
    
    Parameters:
    image: numpy.ndarray - Input image (height, width, channels)
    border_width: int - Width of the border in pixels
    border_color: tuple - RGB color of the border (default: red)
    
    Returns:
    numpy.ndarray - Image with added border
    """
    h, w = image.shape[:2]
    
    # Create border mask
    result = image.copy()
    
    # Top border
    result[:border_width, :] = border_color
    # Bottom border
    result[h-border_width:, :] = border_color
    # Left border
    result[:, :border_width] = border_color
    # Right border
    result[:, w-border_width:] = border_color
    
    return result


def min_max_normalize(tensor):
    """
    Normalize a PyTorch tensor to the range [0, 1] using min-max normalization.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def transformation_matrix_distance(A, B, mode="geodesic", t_weight=10):
    assert A.shape == B.shape and A.shape[-2:] == (4, 4), "A and B must be (..., 4, 4) tensors"
    eps = 1e-6  # Small value to avoid zero-gradient issues

    if mode == "frobenius":
        # Frobenius norm for rotation + weighted translation distance
        R_dist = torch.norm(A[..., :3, :3] - B[..., :3, :3], dim=(-2, -1))
        T_dist = torch.norm(A[..., :3, 3] - B[..., :3, 3], dim=-1)
        return R_dist + t_weight * T_dist

    elif mode == "geodesic":
        # Extract rotation matrices
        R1, R2 = A[..., :3, :3], B[..., :3, :3]
        R = torch.matmul(R1.transpose(-2, -1), R2)  # Relative rotation

        # Compute geodesic distance for rotation
        trace = ((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + eps, 1 - eps)
        angle = torch.arccos(trace)  # Angle in radians

        # Compute Euclidean translation distance
        T_dist = torch.norm(A[..., :3, 3] - B[..., :3, 3] + eps, dim=-1)

        return angle + t_weight * T_dist


def gaussian_kernel(size, sigma):
    # Create an array of points from -size//2 to size//2
    x = torch.linspace(-size // 2, size // 2, size)
    # Compute the Gaussian function for each point
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    # Normalize the kernel so that its sum is 1
    return kernel / kernel.sum()



def generate_random_poses(n_poses, n_parts, translation_range, rotation_range):
    '''
    Generate random part poses for training.
    First, generate random translations and rotations.
    Translations are in the range of [-translation_range, translation_range].
    Rotations are in the range of [-rotation_range, rotation_range] in euler angles and converted to quaternion.
    Then, generate xyz and quaternion from these.
    '''
    translations = torch.rand(n_poses, n_parts, 3) * 2 * translation_range - translation_range
    rotations = torch.rand(n_poses, n_parts, 3) * 2 * rotation_range - rotation_range
    rotations = rotations.reshape(-1,3)
    rotations = torch.from_numpy(euler2quat(rotations)).reshape(n_poses, n_parts, 4)
    return torch.cat([translations, rotations], dim=-1)

def euler2quat(rot):
    rotation = SO3.from_rpy_radians(rot[:,0], rot[:,1], rot[:,2]).wxyz
    return rotation


from tqdm import tqdm
def render_camera_view(
                joint_poses,
                joint_rigid_optimizer,
                gs_pipeline,
                global_transformation = False,):
    
    with torch.no_grad():
        gs_pipeline.eval()
        gs_pipeline.model.set_background(torch.ones(3))
        all_poses = joint_poses.clone()
        
        camera = joint_rigid_optimizer.load_init_camera_pose(0)
      
        video_frames = []
        n_frame = len(joint_poses)
        for i in tqdm(range(n_frame)):
            if not global_transformation:
                per_video_poses_no_global = all_poses[i].clone()
                per_video_poses_no_global[joint_rigid_optimizer.root_label] = torch.tensor([0,0,0,1,0,0,0],device=per_video_poses_no_global.device)
                joint_rigid_optimizer.apply_to_model(per_video_poses_no_global)
            else:
                joint_rigid_optimizer.apply_to_model(all_poses[i])

            output = joint_rigid_optimizer.dig_model.get_outputs(camera.to(joint_rigid_optimizer.device))
            video_frame = (output['rgb'].detach().cpu().numpy()*255).astype(np.uint8)
            video_frames.append(video_frame)           
    
    return video_frames