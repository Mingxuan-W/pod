"""
pose predictor model
"""
import torch
import torch.nn as nn
import math
from pod.utils.geo_utils import *
import numpy as np
from pod.transforms.rotation_conversion import *
import torchvision.transforms as transforms
import math

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=256, dtype=torch.float16):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).to(dtype)
    self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x):
    '''
    @x: (B,N,D)
    '''
    return x + self.pe[:, :x.size(1)]

class PosePredictor(torch.nn.Module):
    def __init__(self, num_parts, input_size=(224, 224), freeze_backbone=True, decoder_type = 'vit'):
        super(PosePredictor, self).__init__()
        self.num_parts = num_parts
        # Initialize DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        hidden_dtype = torch.float32
        dino_dim = self.dino.num_features
        hidden_dim = dino_dim
        self.pose_query = nn.Parameter(torch.randn(1, 8, hidden_dim, dtype = hidden_dtype))# [pose, config, register * N] tokens
        
        # Output now predicts position (3) and initial x/y columns (6)
        self.decoder_type = decoder_type
        if self.decoder_type == 'vit':
            self.pos_embedding = PositionalEmbedding(hidden_dim, dtype=hidden_dtype)
            self.trans_projection = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        nn.LayerNorm(normalized_shape=hidden_dim, dtype=hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, 3, dtype = hidden_dtype),
                    )
            self.rot_projection = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        nn.LayerNorm(normalized_shape=hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, 6, dtype = hidden_dtype),
                    )
            self.part_trans_projection = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        nn.LayerNorm(normalized_shape=hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, (self.num_parts)*3, dtype = hidden_dtype),
                    )
            self.part_rot_projection = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        nn.LayerNorm(normalized_shape=hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim, dtype = hidden_dtype),
                        torch.nn.ReLU(),
                        nn.Linear(hidden_dim, (self.num_parts)*6, dtype = hidden_dtype),
                    )
            self.dino_proj = nn.Identity()
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model = hidden_dim, nhead = 12, batch_first=True, dim_feedforward = 4*hidden_dim, dropout=0.1, dtype=hidden_dtype), num_layers=6)

        else:
            assert False, "Unknown decoder type"

        # Handle backbone freezing
        self.set_backbone_frozen(freeze_backbone)
            
        # Define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def set_backbone_frozen(self, freeze=True):
        """Set whether the backbone should be frozen or trainable"""
        for param in self.dino.parameters():
            param.requires_grad = not freeze

    def normalize_vectors(self, position, x_col, y_col):
        """
        Apply Gram-Schmidt orthogonalization to get orthonormal basis vectors
        Returns normalized x, y, z columns and position as 4x4 transform matrix
        """
        batch_size = position.shape[0]
        x = torch.nn.functional.normalize(x_col, dim=1)
        y = y_col
        y = y - torch.sum(y * x, dim=1, keepdim=True) * x
        y = torch.nn.functional.normalize(y, dim=1)
        z = torch.cross(x, y, dim=1)
        transforms = torch.zeros(batch_size, 4, 4, device=position.device)
        transforms[:, :3, 0] = x
        transforms[:, :3, 1] = y
        transforms[:, :3, 2] = z
        transforms[:, :3, 3] = position
        transforms[:, 3, 3] = 1.0
        
        return transforms

    def forward(self, x):
        # x shape: B x 3 x H x W
        x = self.preprocess(x)
        
        # Get DINOv2 features
        features = self.dino.get_intermediate_layers(x)[0]  # B x patches x hidden_size
        return self.forward_from_dino(features)
        
    def per_frame_latent(self, x):
        x = self.preprocess(x)
        features = self.dino.get_intermediate_layers(x)[0]  # B x patches x hidden_size
        if self.decoder_type == 'vit':
        # Project to 9-dim output
            features = self.dino_proj(features)
            features = self.pos_embedding(features)
            # Split into position and initial column vectors
            query = self.pose_query.expand(features.shape[0], -1, -1)
            dec = self.decoder(query, features)
        return dec
    
    def forward_from_dino(self, features,):
        if self.decoder_type == 'vit':
        # Project to 9-dim output
            features = self.dino_proj(features)
            features = self.pos_embedding(features)
            # Split into position and initial column vectors
            query = self.pose_query.expand(features.shape[0], -1, -1)
            dec = self.decoder(query, features)

            position = self.trans_projection(dec[:,0])
            rot = self.rot_projection(dec[:,0])
            x_col = rot[:, :3]
            y_col = rot[:, 3:] 
            obj2cam_transform = self.normalize_vectors(position, x_col, y_col)
            
            # also convert the part poses
            part_position = self.part_trans_projection(dec[:,1]).reshape(-1, self.num_parts, 3)
            part_rot = self.part_rot_projection(dec[:,1]).reshape(-1, self.num_parts, 6)
            part_x_col = part_rot[:, :, :3]
            part_y_col = part_rot[:, :, 3:]
            part_transforms = self.normalize_vectors(part_position.view(-1,3), part_x_col.view(-1,3), part_y_col.view(-1,3)).view(-1, self.num_parts, 4, 4)
        
        return obj2cam_transform, part_transforms


