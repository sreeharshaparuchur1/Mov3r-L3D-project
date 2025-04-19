import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention import RoPE2D, DUSt3RAsymmetricCrossAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SimplerPoseHead(nn.Module):
    def __init__(self, B: int, S: int, Num_patches: int, emb_dim: int, alpha: float = 0.5):
        """
        Args:
            B: Batch size
            S: Number of frames
            Num_patches: Number of patches (P)
            emb_dim: Embedding dimension (D)
            alpha: Weight factor for controlling pose blending (if needed)
        """
        super().__init__()
        self.S = S
        self.alpha = alpha
        self.pose_mlp = nn.Sequential(
            nn.LayerNorm(Num_patches*emb_dim),
            nn.Linear(Num_patches*emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3 for rotation (axis-angle), 3 for translation
        )

    def forward(self, feature):
        """
        Args:
            feature: Tensor of shape (B, S, P, D)
        Returns:
            poses: Tensor of shape (B, S, 4, 4)
        """
        B, S, P, D = feature.shape

        # Average over patches to get per-frame feature
        # frame_feats = feature.mean(dim=2)  # (B, S, D)

        # Flatten B and S to apply MLP
        x = feature.view(B, S, P*D)
        pose_params = self.pose_mlp(x)  # (B, S, 6)

        # Split into rotation and translation
        rot, trans = pose_params[..., :3], pose_params[..., 3:]  # (B, S, 3), (B, S, 3)

        # Convert axis-angle to rotation matrix
        rot_matrix = self.axis_angle_to_matrix(rot)  # (B, S, 3, 3)

        # Form full transformation matrix
        pose_matrix = torch.eye(4, device=feature.device).unsqueeze(0).repeat(B, S, 1, 1)  # (B*S, 4, 4)
        pose_matrix[..., :3, :3] = rot_matrix
        pose_matrix[..., :3, 3] = trans

        return pose_matrix #.view(B, S, 4, 4)

    def axis_angle_to_matrix(self, vec):
        """
        Convert axis-angle to rotation matrix using Rodrigues' formula
        Args:
            vec: Tensor of shape (B, S, 3)
        Returns:
            rot_mat: Tensor of shape (B, S, 3, 3)
        """
        B, S, _ = vec.shape
        theta = torch.norm(vec, dim=2, keepdim=True) + 1e-8  # (B, S, 1)
        axis = vec / theta  # (B, S, 3)

        a = torch.cos(theta / 2)
        sin_half_theta = torch.sin(theta / 2).squeeze(-1)
        b, c, d = -axis[..., 0] * sin_half_theta, -axis[..., 1] * sin_half_theta, -axis[..., 2] * sin_half_theta

        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a.squeeze(-1) * d, a.squeeze(-1) * c, a.squeeze(-1) * b, b * d, c * d

        rot_mat = torch.stack([
            aa.squeeze(-1) + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac),
            2 * (bc - ad),     aa.squeeze(-1) + cc - bb - dd, 2 * (cd + ab),
            2 * (bd + ac),     2 * (cd - ab),     aa.squeeze(-1) + dd - bb - cc,
        ], dim=-1).reshape(B, S, 3, 3)
        
        return rot_mat

    

class PoseHead(nn.Module):
    def __init__(self, B: int, S: int, Num_patches: int, emb_dim: int, alpha: float = 0.5):
        """
        Convert batched depth maps to 3D point clouds.
        Args:
            B (int): Batch size.
            S (int): Sequence length.
            Num_patches (int): Number of patches (from attention mechanism).
            emb_dim (int): Embedding dimension (e.g., 768).
        Returns:
            Pose Estimated (B, S, 4, 4): Pose with respect to the first frame.
        """
        super(PoseHead, self).__init__()

        # Initialize feature tensor
        self.init_features = torch.zeros_like(torch.Tensor(B, S, Num_patches, emb_dim))
        self.similarity_threshold =  alpha

        self.local_state_attention = DUSt3RAsymmetricCrossAttention(emb_dim, emb_dim, depth=1, num_heads = 4)
        self.global_state_attention = DUSt3RAsymmetricCrossAttention(emb_dim, emb_dim, depth=1, num_heads = 4)


    def cosine_similarity(self, features: torch.Tensor, refrence: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between the features for pose estimation.
        Args:
            features (torch.Tensor): Tensor of shape (B, S, Num_patches, emb_dim)
        Returns:
            similarity (torch.Tensor): Cosine similarity between features.
        """
        # Normalize the features along the embedding dimension (dim=-1)
        normalized_features = F.normalize(features, p=2, dim=-1)
        normalized_refrence = F.normalize(refrence, p=2, dim=-1)
        # Compute the cosine similarity (dot product in normalized space)
        sim_per_patch = (normalized_features * normalized_refrence).sum(dim=-1)
        # Average over patches -> (B, S, 1)
        similarity = sim_per_patch.mean(dim=-1, keepdim=True)

        return similarity

    def global_memory_update(self, chunks):
        global_state_memory = {}
        for key, sequence_chunks in tqdm(chunks.items(), desc="Updating Global Memory"):
            # Get Chunks of each batch
            global_state = []
            
            init_global = sequence_chunks[0][0]
            ki =  init_global.view(1,sequence_chunks[0][0].shape[-2],-1).contiguous()
            for i, chunk in enumerate(sequence_chunks):
                #  For each chunk get images
                qry = chunk[0].view(1,chunk[0].shape[-2],-1).contiguous()
                gs  =  self.global_state_attention( qry, ki)
                ki = gs
                global_state.append(gs.expand(chunk.shape[0],-1,-1))

            global_state_memory[key] = torch.cat(global_state,dim=0)        
        
        batch = [a for a in global_state_memory.values()]  # remove the 1 dim
        batch = torch.stack(batch, dim=0)  # [4, 32, 16, 768]
        self.global_memory = batch



    def local_memory_update(self, chunks):
        local_state_memory = {}
        for key, sequence_chunks in tqdm(chunks.items(), desc="Updating Local Memory"):
            # Get Chunks of each batch
            local_state = []
            for chunk in sequence_chunks:
                #  For each chunk get images
                init = chunk[0]
                ki = init.view(1,chunk[0].shape[-2],-1).contiguous()

                for image in range(0, chunk.shape[0]):
                    qry = chunk[image].view(1,chunk[image].shape[-2],-1).contiguous()
                    ls  =  self.local_state_attention( qry, ki)
                    ki = ls
                    local_state.append(ls)
            local_state_memory[key] = torch.stack(local_state,dim=1)
        
        batch = [a.squeeze(0) for a in local_state_memory.values()]  # remove the 1 dim
        batch = torch.stack(batch, dim=0)  # [4, 32, 16, 768]
        self.local_memory = batch

    def make_chunks(self, features, threshold=0.95):
        B, S, P, D = features.shape
        chunk_idxs = []
        sequences_chunks = {}

        for b in range(B):
            sequence = features[b]  # (S, P, D)
            ref = sequence[0]       # First frame is keyframe
            keyframes = [0]

            for t in range(1, S):
                current = sequence[t]
                sim = self.cosine_similarity(
                    current.unsqueeze(0).unsqueeze(0),
                    ref.unsqueeze(0).unsqueeze(0)
                )
                if sim.item() < threshold:
                    keyframes.append(t)
                    ref = current

            # Make sure to include the last frame
            # if keyframes[-1] != S:
            #     keyframes.append(S)

            chunk_idxs.append(keyframes)

        # Now use keyframe indices to chunk sequences
        for b in range(B):
            sequence = features[b]  # (S, P, D)
            keyframes = chunk_idxs[b]
            chunks = []
            if len(keyframes) != 1:
                for i in range(len(keyframes) - 1):
                    start = keyframes[i]
                    end = keyframes[i + 1]
                    chunk = sequence[start:end, :, :]  # (frames, P, D)
                    chunks.append(chunk)
                chunk = sequence[keyframes[i+1]:, :, :]
                chunks.append(chunk)
            else:
                chunk = sequence[:, :, :]  # (frames, P, D)
                chunks.append(chunk)
            sequences_chunks[b] = chunks

        return sequences_chunks, chunk_idxs
    
    def toransform_pose_tokens(self):
        fused_pose = torch.matmul(self.local_memory, self.global_memory)
        return fused_pose
    
    def to_initial_pose(self, local_pose, global_pose):
        """
        Convert local and global pose to initial pose.
        Args:
            local_pose: Local pose tensor of shape (B, S, 4, 4)
            global_pose: Global pose tensor of shape (B, S, 4, 4)
        Returns:
            final_pose: Tensor of shape (B, S, 4, 4)
        """
        # Multiply local and global pose matrices
        final_pose = torch.matmul(local_pose, global_pose)
        # Normalize the pose matrices ??
        return final_pose


if __name__ == "__main__":
    # Simulate feature input with minor noise
    B, S, P, D = 4, 32, 16 ,768   # batch size, sequence length, patches, embedding dim
    # Step 1: Initialize all features to ones
    features = torch.ones(B, S, P, D)

    features[0, 2] = -1
    features[1, 3] = -1

    model =  PoseHead(B, S, P, D)
    chunks, chunk_idxs = model.make_chunks(features)
    model.local_memory_update(chunks)
    model.global_memory_update(chunks)

    # fused_tokens = model.toransform_pose_tokens()

    pose_head = SimplerPoseHead(B, S, P, D)
    import pdb; pdb.set_trace()
    local_pose = pose_head.forward(model.local_memory)
    global_pose = pose_head.forward(model.global_memory)

    final_pose = model.to_initial_pose(local_pose, global_pose)

    print(final_pose.shape)








        
