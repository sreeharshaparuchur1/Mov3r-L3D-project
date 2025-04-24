import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import pdb


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
        return torch.cos(freqs).unsqueeze(1).repeat(1, 1, 2).contiguous()
    
    def _compute_sin_cached(self):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len).type_as(freqs)
        freqs = torch.outer(t, freqs)
        return torch.sin(freqs).unsqueeze(1).repeat(1, 1, 2).contiguous()
    
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
        x_rot = x_half_rot.view(batch_size, seq_len, -1).contiguous()
        
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
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))


        # self.dec_blocks2 = nn.ModuleList([
        #     CrossBlock(decoder_dim, num_heads, 4.0, dropout)
        #     for _ in range(depth)
        # ])
        
    def forward(self, feat_depth, feat_rgb):
        """
        x1: tokens from first view [batch_size, seq_len, encoder_dim]
        x2: tokens from second view [batch_size, seq_len, encoder_dim]
        """
        B = feat_depth.shape[0]
        cls_token = self.cls_token.expand(B, 1, -1)  # [batch_size, 1, decoder_dim]
        feat_depth = torch.cat([cls_token, feat_depth], dim=1)

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
    
class CausalSelfAttentionMasked(nn.Module):
    def __init__(self, encoder_dim, seq_len=None, num_heads=8, dropout=0.0, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = encoder_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.to_k = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.to_v = nn.Linear(encoder_dim, encoder_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Dropout(dropout)
        )

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE2D(encoder_dim // num_heads)
        
        if seq_len is not None:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                    .view(1, 1, seq_len, seq_len))
        else:
            self.mask = None

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (B, S, D)
            mask: (B, S), True for valid, False for pad
        """
        B, S, D = x.shape
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x) # (B, S, D)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) # (B, H, S, D//H)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        assert not torch.isnan(q).any(), "NaN in q"
        assert not torch.isnan(k).any(), "NaN in k"
        assert not torch.isnan(v).any(), "NaN in v"

        if self.use_rope:
            pos_q = torch.arange(S, device=x.device)
            pos_k = torch.arange(S, device=x.device)
            q = rearrange(q, 'b h n d -> (b h) n d')
            k = rearrange(k, 'b h n d -> (b h) n d')
            q = self.rope(q, pos_q)
            k = self.rope(k, pos_k)
            q = rearrange(q, '(b h) n d -> b h n d', h=self.num_heads)
            k = rearrange(k, '(b h) n d -> b h n d', h=self.num_heads)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.mask is not None:
            attn = attn.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
        else:
            # Causal mask created on the fly for global memory
            mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0).to(x.device)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        # Apply padding mask
        if pad_mask is not None:
            pad_mask = pad_mask.to(torch.float) # 
            pad_mask = pad_mask.unsqueeze(-2)
            pad_mask = torch.matmul(pad_mask.permute(0,2,1), pad_mask).unsqueeze(1) # B, 1, S, S
            attn = attn.masked_fill(pad_mask == 0, float('-inf')).to(x.device)


        # Handle rows with all -inf
        inf_mask = torch.isinf(attn)
        all_inf = inf_mask.all(dim=-1, keepdim=True)
        
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(all_inf, 0.0) #check for attn[7,0]
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        assert torch.all(torch.isfinite(out)), "self_attn output is not finite"
        return self.to_out(out)

class CausalSelfAttentionBlockMasked(nn.Module):
    def __init__(self, encoder_dim, seq_len=None, num_heads=8, dropout=0.0, mlp_ratio =4, use_rope=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.self_attn = CausalSelfAttentionMasked(encoder_dim, seq_len, num_heads, dropout=dropout, use_rope=use_rope)

        self.norm2 = nn.LayerNorm(encoder_dim)
        mlp_hidden_dim = int(encoder_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, encoder_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pad_mask=None):
        residual = x
        x = self.norm1(x)
        x_attn = self.self_attn(x, pad_mask)
        x = residual + x_attn

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x        

class CausalSelfAttention_masked(nn.Module):
    def __init__(self, encoder_dim, seq_len=None, depth=4, num_heads=8, dropout=0.0, use_rope=False):
        super().__init__()
        self.dec_blocks = nn.ModuleList([
            CausalSelfAttentionBlockMasked(encoder_dim, seq_len, num_heads, dropout, use_rope=use_rope)
            for _ in range(depth)
        ])

    def forward(self, x, pad_mask=None):
        #TODO linear proj here?
        for block in self.dec_blocks:
            try:
                x = block(x, pad_mask)
            except Exception as e:
                print("NAN aagaya!!!")
                print(f"Caught exception: {e}")
                pdb.set_trace()
                x = block(x, pad_mask)
        return x
