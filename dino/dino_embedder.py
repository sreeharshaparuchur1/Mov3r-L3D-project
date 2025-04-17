import torch
import torch.nn as nn

from dino.layers.patch_embed import PatchEmbed
from dino.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


class DINOEmbedder(nn.Module):
    
    def __init__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens=0,#TODO: 4?
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        super().__init__()
        
        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            interpolate_antialias,
            interpolate_offset,
            block_chunks,
            init_values,
            embed_dim
        )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, x):
        """
        Forward pass through the patch embed layer.
        """
        patch_tokens = self.patch_embed(x)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        return patch_tokens
    
if __name__ == "__main__":
    rgb= torch.randn(1, 3, 224, 224).cuda()
    patch_embed = 'dinov2_vitl14_reg'
    img_size = 224
    patch_size = 16

    dino_embedder = DINOEmbedder(
        patch_embed,
        img_size,
        patch_size,
    ).cuda()

    dino_embedder.eval()
    with torch.no_grad():
        output = dino_embedder(rgb)
        print(output.shape)  # Should be (1, 1024, 14, 14) for a 224x224 image with 16x16 patches