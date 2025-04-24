import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention import CausalSelfAttention_masked

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from heads.utils_pose import quaternion_multiply, rotate_vector_by_quaternion, activate_pose, quat_to_mat, mat_to_quat

class SimplerPoseHead(nn.Module):
    def __init__(self, S: int, emb_dim: int, alpha: float = 0.5):
        """
        Args:
            B: Batch size
            C: Number of chunks
            S: Number of frames
            emb_dim: Embedding dimension (D)
            alpha: Weight factor for controlling pose blending (if needed)
        """
        super().__init__()
        self.S = S
        self.alpha = alpha
        self.pose_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 3 for rotation (axis-angle), 3 for translation
        )

    def forward(self, feature):
        """
        Args:
            feature: Tensor of shape (B, S, P, D)
        Returns:
            poses: Tensor of shape (B, S, 7)
        """
        B, C, S, D = feature.shape

        # Average over patches to get per-frame feature
        # frame_feats = feature.mean(dim=2)  # (B, S, D)
        
        # Apply MLP to get pose parameters
        pose_params = self.pose_mlp(feature)  # (B, C, S, 6)

        # # Split into rotation and translation
        # rot, trans = pose_params[..., :3], pose_params[..., 3:]  # (B, C, S, 3), (B, C, S, 3)
        
        # # Convert axis-angle to rotation matrix
        # rot_matrix = self.axis_angle_to_matrix(rot)  # (B, C, S, 3, 3)

        # # Form full transformation matrix
        # pose_matrix = torch.eye(4, device=feature.device).unsqueeze(0).repeat(B, C, S, 1, 1)  # (B*C*S, 4, 4)
        # pose_matrix[..., :3, :3] = rot_matrix
        # pose_matrix[..., :3, 3] = trans

        # return pose_matrix #.view(B, S, 4, 4)
        return pose_params

    def axis_angle_to_matrix(self, vec):
        """
        Convert axis-angle to rotation matrix using Rodrigues' formula
        Args:
            vec: Tensor of shape (B, C, S, 3)
        Returns:
            rot_mat: Tensor of shape (B, C, S, 3, 3)
        """
        B, C, S, _ = vec.shape
        theta = torch.norm(vec, dim=-1, keepdim=True) + 1e-8  # (B, C, S, 1)
        axis = vec / theta  # (B, C, S, 3)
        
        half_theta = theta / 2
        a = torch.cos(half_theta).squeeze(-1)
        sin_half_theta = torch.sin(theta / 2).squeeze(-1)
        b = -axis[..., 0] * sin_half_theta
        c = -axis[..., 1] * sin_half_theta 
        d = -axis[..., 2] * sin_half_theta

        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        rot_mat = torch.stack([
            aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac),
            2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab),
            2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc,
        ], dim=-1).reshape(B, C, S, 3, 3)
        
        return rot_mat

    

class PoseHead(nn.Module):
    def __init__(self, S: int, emb_dim: int, alpha: float = 0.5):
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

        self.similarity_threshold =  alpha

        self.local_state_attention = CausalSelfAttention_masked(emb_dim, seq_len=S, depth=4, num_heads = 4) #seq_len is padded chunk size 
        self.global_state_attention = CausalSelfAttention_masked(emb_dim, depth=4, num_heads = 4) #mask created on the fly
        self.global_memory = None
        self.local_memory = None

        self.local_pose_head = SimplerPoseHead(S, emb_dim)
        self.global_pose_head = SimplerPoseHead(1, emb_dim)

    def cosine_similarity(self, features: torch.Tensor, refrence: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between the features for pose estimation.
        Args:
            features (torch.Tensor): Tensor of shape (B, S, emb_dim)
        Returns:
            similarity (torch.Tensor): Cosine similarity between features.
        """
        # Normalize the features along the embedding dimension (dim=-1)
        normalized_features = F.normalize(features, p=2, dim=-1)
        normalized_refrence = F.normalize(refrence, p=2, dim=-1)
        # Compute the cosine similarity (dot product in normalized space)
        similarity = (normalized_features * normalized_refrence).sum(dim=-1)

        return similarity

    def global_memory_update(self, chunks, chunk_mask):
        """
        Update global memory using the global state attention mechanism.
        For each chunk, the first frame is used as the keyframe.
        The global state attention applied to all keyframes causally.
        Args:
            chunks (torch.Tensor): Tensor of shape (B, C, S, D) containing chunked sequences.
            Returns: 
                None
        """

        B, C, S, D = chunks.shape #using cls_token, P = 1
        # extract the first frame of each chunk as the keyframe
        keyframes = chunks[:, :, 0, :]  # (B, C, D)
        keyframe_mask = chunk_mask[:, :, 0]  # (B, C)

        attended = self.global_state_attention(keyframes, pad_mask=keyframe_mask)  # (B, C, D)
        
        self.global_memory = attended.unsqueeze(2)  # (B, C, 1, D)


    def local_memory_update(self, chunks, chunk_mask):
        """
        Update local memory using the local state attention mechanism.
        For each chunk, the first frame is used as the keyframe.
        The local state attention is applied to each chunk, and the results are stored in local_state_memory.
        Args:
            chunks (torch.Tensor): Tensor of shape (B, C, S, D) containing chunked sequences.
            chunk_mask (torch.Tensor): Tensor of shape (B, C, S) containing masks for the chunks.
        Returns:
            None
        """
        B, C, S, D = chunks.shape #using cls_token, P = 1
        
        # Flatten batch and chunk dimensions
        chunks_flat = chunks.view(B * C, S, D).contiguous()  # (B*C, S, D)
        chunk_mask_flat = chunk_mask.view(B * C, S).contiguous()  # (B*C, S)

        attended = self.local_state_attention(chunks_flat, pad_mask=chunk_mask_flat)  # (B*C, S, D)
        
        # Reshape back to (B, C, S, D)
        self.local_memory = attended.view(B, C, S, D).contiguous()  # (B, C, S, D)

        # # Keyframe = first frame in each chunk
        # keyframes = chunks_flat[:, 0, :, :]         # (B*C, P, D)
        # seq = chunks_flat.view(B * C, S * P, D)      # (B*C, S*P, D)
        # mask = chunk_mask_flat.unsqueeze(-1).expand(-1, -1, P)  # (B*C, S, P)
        # mask = mask.reshape(B * C, S * P)            # (B*C, S*P), bool

        # # mask for causal attention and OR with chunk mask
        

        # # Pass through local attention (query = keyframe, key/value = entire chunk)
        # attended = self.local_state_attention(keyframes, seq, mask)  # (B*C, P, D)

        # # Reshape back to (B, C, P, D), then unsqueeze to (B, C, 1, P, D) as one keyframe per chunk
        # local_memory = attended.view(B, C, P, D).unsqueeze(2)  # (B, C, 1, P, D)
        # self.local_memory = local_memory  # update class attribute
           

    def make_chunks(self, features, threshold=0.95):
        B, S, D = features.shape
        chunk_idxs = []
        sequences_chunks = []
        num_chunks_list = []
        masks_chunks = []

        for b in range(B):
            sequence = features[b]  # (S, D)
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

        # Now use keyframe indices to chunk sequences and pad to max length S
        for b in range(B):
            sequence = features[b]         # (S_total, D)
            keyframes = chunk_idxs[b]
            chunks = []
            masks = []

            if len(keyframes) != 1:
                for i in range(len(keyframes) - 1):
                    start = keyframes[i]
                    end = keyframes[i + 1]
                    chunk = sequence[start:end, :]  # (T, D)
                    T = chunk.shape[0]

                    if T < S:
                        pad_len = S - T
                        chunk = F.pad(chunk, (0, 0, 0, pad_len))  # Pad to (S, D)
                        mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
                    else:
                        chunk = chunk[:S]
                        mask = torch.ones(S)

                    chunks.append(chunk)
                    masks.append(mask)

                # Last chunk
                chunk = sequence[keyframes[-1]:, :] # (T, D)
                T = chunk.shape[0]
                if T < S:
                    pad_len = S - T
                    chunk = F.pad(chunk, (0, 0, 0, pad_len))
                    mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
                else:
                    chunk = chunk[:S]
                    mask = torch.ones(S)

                chunks.append(chunk)
                masks.append(mask)
            else:
                chunk = sequence[:, :] # (T, D)
                T = chunk.shape[0]
                if T < S:
                    pad_len = S - T
                    chunk = F.pad(chunk, (0, 0, 0, pad_len))
                    mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
                else:
                    chunk = chunk[:S]
                    mask = torch.ones(S)
                chunks.append(chunk)
                masks.append(mask)

            sequences_chunks.append(torch.stack(chunks))     # (C_i, S, D)
            masks_chunks.append(torch.stack(masks))          # (C_i, S)
            num_chunks_list.append(len(chunks))

        max_chunks = max(num_chunks_list)

        padded_chunks = []
        padded_masks = []

        for chunks, masks in zip(sequences_chunks, masks_chunks):
            C_i = chunks.shape[0]

            if C_i < max_chunks:
                chunk_pad = torch.zeros((max_chunks - C_i, S, D), device=chunks.device, dtype=chunks.dtype)
                mask_pad = torch.zeros((max_chunks - C_i, S), device=masks.device, dtype=masks.dtype)
                chunks = torch.cat([chunks, chunk_pad], dim=0)  # (max_chunks, S, D)
                masks = torch.cat([masks, mask_pad], dim=0)     # (max_chunks, S)
            else:
                chunks = chunks[:max_chunks]
                masks = masks[:max_chunks]

            padded_chunks.append(chunks)
            padded_masks.append(masks)

        padded_chunks = torch.stack(padded_chunks)     # (B, C, S, D)
        chunk_mask = torch.stack(padded_masks).bool().to(features.device)  # (B, C, S)

        return padded_chunks, chunk_mask, chunk_idxs
    
    def unchunk(self, chunked_pose, chunk_idxs):
        """
        Remove padding and Unchunk the tensor back to the original sequence length.
        Args:
            tensor: Tensor of shape (B, C, S, 7)
            chunk_idxs: List of chunk indices for each batch
        Returns:
            unchunked_tensor: Tensor of shape (B, S, 7)
        """
        B, C, S, _ = chunked_pose.shape
        unchunked_tensor = torch.zeros(B, S, 7).to(chunked_pose.device)

        for b in range(B):
            for c in range(C):
                if c >= len(chunk_idxs[b]):
                    break
                start = chunk_idxs[b][c]
                end = chunk_idxs[b][c + 1] if c + 1 < len(chunk_idxs[b]) else S
                
                unchunked_tensor[b, start:end] = chunked_pose[b, c, start:end]

        return unchunked_tensor
    
    def rt_to_pose(self, R, trans):
        B, C, S, _ = trans.shape

        R = torch.cat([R, trans.unsqueeze(-1)], dim=-1)
        last_row = torch.tensor([0,0,0,1]).to(R.device)
        last_row = last_row.repeat(B, C, S, 1).unsqueeze(-2)
        R = torch.cat([R, last_row], dim=-2)

        return R

    def to_initial_pose(self, local_pose, global_pose):
        """
        Convert local and global pose to initial pose.
        Args:
            local_pose: Local pose tensor of shape (B, C, S, 7)
            global_pose: Global pose tensor of shape (B, C, 1, 7)
        Returns:
            final_pose: Tensor of shape (B, C, S, 7)
        """
        _, _, S, _ = local_pose.shape
        global_pose = global_pose.repeat( 1, 1, S, 1)
        quat_l, trans_l = F.normalize(local_pose[..., :4], dim=-1), local_pose[..., 4:]
        quat_g, trans_g = F.normalize(global_pose[..., :4], dim=-1), global_pose[..., 4:]

        # Multiply local and global pose quaternions
        # quat_f = quaternion_multiply(quat_g.expand_as(quat_l), quat_l)
        # trans_f_rotated = rotate_vector_by_quaternion(trans_l, quat_g.expand(-1, -1, trans_l.shape[2], -1))
        # trans_f = trans_g.expand_as(trans_f_rotated) + trans_f_rotated
        # final_pose = torch.cat([quat_f, trans_f], dim=-1)
        
        R_l, R_g = quat_to_mat(quat_l), quat_to_mat(quat_g)
        T_l = self.rt_to_pose(R_l, trans_l)
        T_g = self.rt_to_pose(R_g, trans_g)

        T_f = torch.matmul(T_l, T_g)
        final_pose = torch.cat([mat_to_quat(T_f[..., :3, :3]), T_f[..., -1,: 3]], dim=-1)

        return final_pose
    
        #[u,v,1]         = K R_c<-w [Xw, Yw, Zw, 1]
        #[u,v,1]         = K R_c<-k  R_k<-w [Xw, Yw, Zw, 1]

    

    def forward(self, features):
        """
        Forward pass for the PoseHead.
        Args:
            features: Tensor of shape (B, S, D)
        Returns:
            final_pose: Tensor of shape (B, S, 7)
        """
        # import pdb; pdb.set_trace()
        # Step 1: Chunk the features
        padded_chunks, chunk_mask, chunk_idxs = self.make_chunks(features)

        # Step 2: Update local and global memory
        self.local_memory_update(padded_chunks, chunk_mask)
        self.global_memory_update(padded_chunks, chunk_mask)
        # Step 3: Transform pose tokens
        local_pose = self.local_pose_head(self.local_memory)  # (B, C, S, 7)
        global_pose = self.global_pose_head(self.global_memory)  # (B, C, 1, 7)
        # Step 4: Convert to initial pose
        final_pose = self.to_initial_pose(local_pose, global_pose)  # (B, C, S, 7)
        # Step 5: Unchunk the final pose
        final_pose = self.unchunk(final_pose, chunk_idxs)  # (B, S, 7)
        final_activated_pose = activate_pose(final_pose, act_type="relu")
        return final_pose

if __name__ == "__main__":
    # Simulate feature input with minor noise
    B, S, D = 2, 5 ,768   # batch size, sequence length, patches, embedding dim
    # Step 1: Initialize all features to ones
    features = torch.ones(B, S, D).to('cuda')

    features[0, 2] = -1
    features[1, 3] = -1
    starttime = time.time()
    # model =  PoseHead(B, S, D).to('cuda')
    
    # padded_chunks, chunk_mask, chunk_idxs = model.make_chunks(features)
    # model.local_memory_update(padded_chunks, chunk_mask) #(B, C, S, D)
    # model.global_memory_update(padded_chunks, chunk_mask) #(B, C, 1, D)

    # # fused_tokens = model.toransform_pose_tokens()
    # B, C, S, D = model.local_memory.shape
    # local_pose_head = SimplerPoseHead(B, S, D).to('cuda')
    # global_pose_head = SimplerPoseHead(B, 1, D).to('cuda')
    
    # local_pose = local_pose_head.forward(model.local_memory) # (B, C, S, 7)
    # global_pose = global_pose_head.forward(model.global_memory) # (B, C, 1, 7)

    # final_pose = model.to_initial_pose(local_pose, global_pose) # (B, C, S, 7)

    # final_pose = model.unchunk(final_pose, chunk_idxs) # (B, S, 7)
    test_posehead = PoseHead(S, D).to('cuda')
    final_pose = test_posehead(features)
    endtime = time.time()

    print('Job took: ', endtime-starttime)
    print(final_pose.shape)








        
