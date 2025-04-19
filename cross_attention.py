import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding (RoPE) as used in DUSt3R.
    DUSt3R applies RoPE to the queries and keys in cross-attention layers.
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create position encodings
        self.register_buffer(
            "cos_cached", self._compute_cos_cached(), persistent=False
        )
        self.register_buffer(
            "sin_cached", self._compute_sin_cached(), persistent=False
        )
        
    def _compute_cos_cached(self):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len).type_as(freqs)
        freqs = torch.outer(t, freqs)
        return torch.cos(freqs).view(self.max_seq_len, 1, self.dim // 2).repeat(1, 1, 2).contiguous()
    
    def _compute_sin_cached(self):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len).type_as(freqs)
        freqs = torch.outer(t, freqs)
        return torch.sin(freqs).view(self.max_seq_len, 1, self.dim // 2).repeat(1, 1, 2).contiguous()
    
    def forward(self, x, seq_idx):
        # x: [batch_size, seq_len, dim]
        # seq_idx: positional indices
        batch_size, seq_len, _ = x.shape
        
        if seq_idx is None:
            seq_idx = torch.arange(seq_len, device=x.device)
        
        cos = self.cos_cached[seq_idx].transpose(1,0)  # [seq_len, 1, dim]
        sin = self.sin_cached[seq_idx].transpose(1,0)  # [seq_len, 1, dim]
        
        # Split embedding into half for rotation
        x_half = x.view(batch_size, seq_len, -1, 2).contiguous()
        x_half_rot = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1)
        x_rot = x_half_rot.reshape(batch_size, seq_len, -1)
        
        # Apply rotary embeddings
        return x * cos + x_rot * sin


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism as used in DUSt3R.
    Enables information exchange between two views.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Rotary positional embeddings
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE2D(dim // num_heads)
        
    def forward(self, x, context):
        """
        x: tokens from one view [batch_size, seq_len_q, dim]
        context: tokens from another view [batch_size, seq_len_k, dim]
        """
        batch_size, seq_len_q, _ = x.shape
        _, seq_len_k, _ = context.shape
        
        # Get query, key, value projections
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Apply rotary positional embeddings if enabled
        if self.use_rope:
            q_pos = torch.arange(seq_len_q, device=q.device)
            k_pos = torch.arange(seq_len_k, device=k.device)
            
            # Apply RoPE to queries and keys
            q = rearrange(q, 'b h n d -> (b h) n d')
            k = rearrange(k, 'b h n d -> (b h) n d')
            
            q = self.rope(q, q_pos)
            k = self.rope(k, k_pos)
            
            q = rearrange(q, '(b h) n d -> b h n d', h=self.num_heads)
            k = rearrange(k, '(b h) n d -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Final projection
        return self.to_out(out)


class CrossBlock(nn.Module):
    """
    CrossBlock as used in DUSt3R decoder.
    Applies self-attention, followed by cross-attention, and an MLP.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout, use_rope)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context):
        # Self attention
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + x_attn
        
        # Cross attention
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(x, context)
        
        # MLP
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)
        
        return x


class PatchWiseCrossAttentionDecoder(nn.Module):
    """
    Complete decoder implementation with cross-attention as used in DUSt3R.
    """
    def __init__(self, dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0, use_rope=True):
        super().__init__()
        self.depth = depth
        
        # Decoder embedding projection
        self.decoder_embed = nn.Linear(1024, dim)  # From encoder dim to decoder dim
        
        # Decoder blocks with cross-attention
        self.dec_blocks = nn.ModuleList([
            CrossBlock(dim, num_heads, mlp_ratio, dropout, use_rope)
            for _ in range(depth)
        ])
        
    def forward(self, x1, x2):
        """
        x1: tokens from first view [batch_size, seq_len, encoder_dim]
        x2: tokens from second view [batch_size, seq_len, encoder_dim]
        """
        # Project to decoder dimension
        x1 = self.decoder_embed(x1)
        x2 = self.decoder_embed(x2)
        
        # Apply decoder blocks with cross-attention
        for block in self.dec_blocks:
            x1 = block(x1, x2)  # View 1 attends to View 2
            x2 = block(x2, x1)  # View 2 attends to View 1
            
        return x1, x2


# class DUSt3RAsymmetricCrossAttention(nn.Module):
#     """
#     DUSt3R's asymmetric cross-attention implementation with two decoders.
#     """
#     def __init__(self, encoder_dim=1024, decoder_dim=768, depth=12, num_heads=12, dropout=0.0):
#         super().__init__()
        
#         # Input to decoder projection
#         self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        
#         # Two separate decoders with CrossBlocks
#         self.dec_blocks1 = nn.ModuleList([
#             CrossBlock(decoder_dim, num_heads, 4.0, dropout)
#             for _ in range(depth)
#         ])
        
#         self.dec_blocks2 = nn.ModuleList([
#             CrossBlock(decoder_dim, num_heads, 4.0, dropout)
#             for _ in range(depth)
#         ])
        
#     def forward(self, x1, x2):
#         """
#         x1: tokens from first view [batch_size, seq_len, encoder_dim]
#         x2: tokens from second view [batch_size, seq_len, encoder_dim]
#         """
#         # Project to decoder dimension
#         x1 = self.decoder_embed(x1)
#         x2 = self.decoder_embed(x2)
        
#         # Apply decoder blocks with cross-attention
#         for block1, block2 in zip(self.dec_blocks1, self.dec_blocks2):
#             x1_new = block1(x1, x2)  # View 1 attends to View 2
#             x2_new = block2(x2, x1)  # View 2 attends to View 1
#             x1, x2 = x1_new, x2_new
            
#         return x1, x2
    
class DUSt3RAsymmetricCrossAttention(nn.Module):
    """
    DUSt3R's asymmetric cross-attention implementation with two decoders.
    """
    def __init__(self, encoder_dim=1024, decoder_dim=768, depth=12, num_heads=12, dropout=0.0):
        super().__init__()
        
        # Input to decoder projection
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        
        # Two separate decoders with CrossBlocks
        self.dec_blocks1 = nn.ModuleList([
            CrossBlock(decoder_dim, num_heads, 4.0, dropout)
            for _ in range(depth)
        ])
        
        # self.dec_blocks2 = nn.ModuleList([
        #     CrossBlock(decoder_dim, num_heads, 4.0, dropout)
        #     for _ in range(depth)
        # ])
        
    def forward(self, feat_depth, feat_rgb):
        """
        x1: tokens from first view [batch_size, seq_len, encoder_dim]
        x2: tokens from second view [batch_size, seq_len, encoder_dim]
        """
        # Project to decoder dimension
        x1 = self.decoder_embed(feat_depth)
        x2 = self.decoder_embed(feat_rgb)
        
        # Apply decoder blocks with cross-attention
        for block1 in self.dec_blocks1:
            x1_new = block1(x1, x2)  # View 2 attends to View 1
            x1 = x1_new
        return x1

class CrossAttentionMasked(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE2D(dim // num_heads)

    def forward(self, x, context, context_mask=None):
        """
        Args:
            x: (B, Nq, D)
            context: (B, Nk, D)
            context_mask: (B, Nk), True for valid, False for pad
        """
        B, Nq, D = x.shape
        _, Nk, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.use_rope:
            pos_q = torch.arange(Nq, device=x.device)
            pos_k = torch.arange(Nk, device=x.device)
            q = rearrange(q, 'b h n d -> (b h) n d')
            k = rearrange(k, 'b h n d -> (b h) n d')
            q = self.rope(q, pos_q)
            k = self.rope(k, pos_k)
            q = rearrange(q, '(b h) n d -> b h n d', h=self.num_heads)
            k = rearrange(k, '(b h) n d -> b h n d', h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)

        #TODO: add causal mask
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nk)
            attn = attn.masked_fill(~context_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, Nq, D)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossBlockMasked(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttentionMasked(dim, num_heads, dropout, use_rope)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, context_mask=None):
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + x_attn

        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(x, context, context_mask)

        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)

        return x


class DUSt3RAsymmetricCrossAttention_masked(nn.Module):
    def __init__(self, encoder_dim=1024, decoder_dim=768, depth=12, num_heads=12, dropout=0.0):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.dec_blocks1 = nn.ModuleList([
            CrossBlockMasked(decoder_dim, num_heads, 4.0, dropout)
            for _ in range(depth)
        ])

    def forward(self, feat_depth, feat_rgb, mask_rgb=None):
        x1 = self.decoder_embed(feat_depth)
        x2 = self.decoder_embed(feat_rgb)

        for block1 in self.dec_blocks1:
            x1_new = block1(x1, x2, mask_rgb)
            x1 = x1_new

        return x1