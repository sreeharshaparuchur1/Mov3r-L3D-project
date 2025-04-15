import torch


import torch
import torch.nn.functional as F

import torch

def ConfAlignPointMapRegLoss(gt_batch, prediction, alpha=0.01, eps=1e-6):
    """
    Confidence-weighted alignment regression loss.

    Args:
        gt_batch: Tensor of shape [B, K, H, W, 3], ground truth pointmaps (x, y, z).
        prediction: Tensor of shape [B, K, H, W, C], contains:
                    prediction[..., 0:3] = x̂ (predicted pointmap [x, y, z])
                    prediction[..., 4]   = c (confidence)
        alpha: Weight for confidence regularization term.
        eps: Small constant to avoid division by zero and log(0).
    
    Returns:
        Scalar loss.
    """

    x_hat = prediction[..., 0:3]       # Predicted 3D pointmap
    c = prediction[..., 4:5]           # Confidence, keep dimensions [B, K, H, W, 1]

    gt = gt_batch                      # Ground truth 3D pointmap

    # Assuming scale is 1, alignment term becomes simple L2 diff weighted by confidence
    diff = x_hat - gt                  # [B, K, H, W, 3]
    loss_data = c * (diff ** 2)       # Confidence-weighted squared error

    # Confidence regularization: -α log(c)
    conf_reg = -alpha * torch.log(c + eps)

    total_loss = loss_data + conf_reg # [B, K, H, W, 3]
    return total_loss.mean()



def ConfAlignDepthRegLoss(gt_batch, prediction, alpha=0.01, eps=1e-6):
    """
    Confidence-weighted alignment regression loss.

    Args:
        gt_batch: Tensor of shape [B, K, H, W, 1], ground truth Depth (d).
        prediction: Tensor of shape [B, K, H, W, C], contains:
                    prediction[..., 0] = x̂ (predicted Depth [d])
                    prediction[..., 1]   = c (confidence)
        alpha: Weight for confidence regularization term.
        eps: Small constant to avoid division by zero and log(0).
    
    Returns:
        Scalar loss.
    """

    x_hat = prediction[..., 0].unsqueeze(-1)       # Predicted Depth
    c = prediction[..., 1].unsqueeze(-1)            # Confidence, keep dimensions [B, K, H, W, 1]

    gt = gt_batch                      # Ground truth Depth

    # Assuming scale is 1, alignment term becomes simple L2 diff weighted by confidence
    diff = x_hat - gt                  # [B, K, H, W, 1]
    loss_data = c * (diff ** 2)       # Confidence-weighted squared error

    # Confidence regularization: -α log(c)
    conf_reg = -alpha * torch.log(c + eps)

    total_loss = loss_data + conf_reg # [B, K, H, W, 1]
    return total_loss.mean()


def test_losses():
    B, K, H, W = 2, 3, 64, 64

    # ----- Test ConfAlignPointMapRegLoss -----
    gt_pointmap = torch.rand(B, K, H, W, 3)
    pred_pointmap = torch.rand(B, K, H, W, 5)
    pred_pointmap[..., 4] = torch.clamp(pred_pointmap[..., 4], min=0.01)  # ensure confidence isn't zero

    loss_pointmap = ConfAlignPointMapRegLoss(gt_pointmap, pred_pointmap)
    print("PointMap Loss:", loss_pointmap.item())

    # ----- Test ConfAlignDepthRegLoss -----
    gt_depth = torch.rand(B, K, H, W, 1)
    pred_depth = torch.rand(B, K, H, W, 2)
    pred_depth[..., 1] = torch.clamp(pred_depth[..., 1], min=0.01)  # ensure confidence isn't zero

    loss_depth = ConfAlignDepthRegLoss(gt_depth, pred_depth)
    print("Depth Loss:", loss_depth.item())

# test_losses()
