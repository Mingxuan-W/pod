import torch
import torch.nn.functional as F
import torch.nn as nn
from pod.transforms.rotation_conversion import rotation_6d_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d

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

def quatmul(q0:torch.Tensor,q1:torch.Tensor):
    w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    return torch.stack(
            [
                -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            ],
            dim = -1
        )

def pose2mat(pose:torch.Tensor):
    """
    Converts a 7-vector pose to a 4x4 transformation matrix
    """
    t = pose[:,:3]
    q = pose[:,3:]
    rot = quat_to_rotmat(q)
    mat = torch.eye(4)[None].repeat(pose.size(0),1,1)
    mat[:,:3,:3] = rot
    mat[:,:3,3] = t
    return mat


def pose2mat_fast(pose:torch.Tensor, mat):
    """
    Converts a 7-vector pose to a 4x4 transformation matrix
    """
    t = pose[:,:3]
    q = pose[:,3:]
    rot = quat_to_rotmat(q)
    mat = torch.eye(4)[None].repeat(pose.size(0),1,1)
    mat[:,:3,:3] = rot
    mat[:,:3,3] = t
    return mat

def clamp_degrees(degrees, min_degree=-180, max_degree=180):
    """
    Clamp the degrees to be within the specified range.
    
    Parameters:
    degrees (torch.Tensor): A tensor of degrees to be clamped.
    min_degree (float): Minimum degree value.
    max_degree (float): Maximum degree value.
    
    Returns:
    torch.Tensor: Clamped degrees.
    """
    min_radians = torch.deg2rad(torch.tensor(min_degree, dtype=degrees.dtype,device=degrees.device))
    max_radians = torch.deg2rad(torch.tensor(max_degree, dtype=degrees.dtype,device=degrees.device))
    return torch.clamp(degrees, min_radians, max_radians)

def axis_angle_to_quaternion(axis, angle):
    """
    Convert an axis-angle rotation to a quaternion in a differentiable way.
    
    Parameters:
    axis (torch.Tensor): A tensor of shape (n, 3) representing the axis of rotation.
    angle (torch.Tensor): A tensor of shape (n, 1) representing the angle of rotation in radians.
    
    Returns:
    torch.Tensor: A tensor of shape (n, 4) representing the quaternion.
    """
    # Normalize the axis to ensure it has unit length
    axis = axis / axis.norm(dim=1, keepdim=True)
    
    # Calculate the quaternion components
    half_angle = angle / 2
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)

    q_x = axis[:, 0] * sin_half_angle.squeeze()
    q_y = axis[:, 1] * sin_half_angle.squeeze()
    q_z = axis[:, 2] * sin_half_angle.squeeze()
    q_w = cos_half_angle.squeeze()

    # Concatenate the components to form the quaternion
    quaternion = torch.stack([q_w, q_x, q_y, q_z], dim=1)

    return quaternion

def standardize_axes(axis_list,degrees_list):
    standardized_axes = axis_list
    standardized_degrees = degrees_list
    for i in range(axis_list.shape[0]):  
        axis = axis_list[i]
        degree = degrees_list[i]
        factor = torch.sign(axis[-1])
        if degree < 0 :
            axis = -axis
            degree = -degree
        standardized_axes[i] = axis
        standardized_degrees[i] = degree   
    return standardized_axes,standardized_degrees

def quaternion_multiply(q1:torch.Tensor, q2:torch.Tensor):
    """
    Multiply two quaternions q1 and q2.
    Args:
    - q1: Tensor of shape (4,) representing the first quaternion (q1_w, q1_x, q1_y, q1_z).
    - q2: Tensor of shape (4,) representing the second quaternion (q2_w, q2_x, q2_y, q2_z).

    Returns:
    - Tensor of shape (4,) representing the product quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.tensor([w, x, y, z], dtype=q1.dtype, device=q1.device)

def quaternions_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    
    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a batch of quaternions to rotation matrices.
    quaternion: [batch_size, 4]
    return: [batch_size, 3, 3]
    """
    batch_size = quaternion.size(0)
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).to(quaternion.device)
    
    rotation_matrix[:, 0, 0] = ww + xx - yy - zz
    rotation_matrix[:, 0, 1] = 2 * (xy - zw)
    rotation_matrix[:, 0, 2] = 2 * (xz + yw)
    rotation_matrix[:, 1, 0] = 2 * (xy + zw)
    rotation_matrix[:, 1, 1] = ww - xx + yy - zz
    rotation_matrix[:, 1, 2] = 2 * (yz - xw)
    rotation_matrix[:, 2, 0] = 2 * (xz - yw)
    rotation_matrix[:, 2, 1] = 2 * (yz + xw)
    rotation_matrix[:, 2, 2] = ww - xx - yy + zz

    return rotation_matrix

def quaternion_mse_loss(q1, q2):
    """
    Calculate the MSE loss between two batches of quaternions.
    q1, q2: [batch_size, 4]
    return: scalar loss
    """
    r1 = quaternion_to_rotation_matrix(q1)
    r1 = matrix_to_rotation_6d(r1)
    r2 = quaternion_to_rotation_matrix(q2)
    r2 = matrix_to_rotation_6d(r2)
    
    mse_loss = nn.MSELoss()
    loss = mse_loss(r1, r2)
    
    return loss

def rot_6d_to_quat(rot_6d):
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rotation_6d_to_matrix(rot_6d)
    quat = matrix_to_quaternion(rot_mat) #N,  wxyz
    return quat

def quat_to_rot_6d(quat):
    """
    Convert quaternion to 6d representation
    quat: N, wxyz
    """
    rot_mat = quaternion_to_rotation_matrix(quat)
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d

def translation_rotation_to_4x4_matrix(translation, rotation):
    """
    Combines a rotation matrix and a translation vector into a 4x4 transformation matrix.

    Args:
        translation (torch.Tensor): A [n,3] translation vector.
        rotation (torch.Tensor): A [n, 3, 3] rotation matrix.

    Returns:
        torch.Tensor: A [n,4, 4] transformation matrix.
    """
    assert translation.size(1) == 3
    assert rotation.shape[1:] == (3, 3)
    n = rotation.size(0)
    transformation = torch.eye(4, device=rotation.device).unsqueeze(0).repeat(n, 1, 1)
    transformation[:, :3, :3] = rotation
    transformation[:, :3, 3] = translation
    return transformation


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles



def mat2pose(mat:torch.Tensor):
    """
    Converts a 4x4 transformation matrix to a 7-vector pose
    """
    rot = mat[:,:3,:3]
    t = mat[:,:3,3]
    q = matrix_to_quaternion(rot)
    return torch.cat([t,q],dim=-1)
