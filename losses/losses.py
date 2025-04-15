import torch


import torch
import torch.nn.functional as F
import torch

def ConfAlignPointMapRegLoss(gt_batch, prediction, intrinsics, alpha=0.01, eps=1e-6):
    """
    Confidence-weighted alignment regression loss.

    Args:
        gt_batch: Tensor of shape [B, K, H, W, 1], ground truth Depth (d).
        prediction: Tensor of shape [B, K, H, W, C], contains:
                    prediction[..., 0:3] = x̂ (predicted Point Map [u,v] -> [x,y,z])
                    prediction[..., 3]   = c (confidence)
        alpha: Weight for confidence regularization term.
        eps: Small constant to avoid division by zero and log(0).
    
    Returns:
        Scalar loss.
    """

    assert prediction.shape[-1] == 4
    assert gt_batch.shape[-1] == 1
    assert intrinsics.shape[-1] == 4
    assert intrinsics.shape[-2] == 3

    B, K, H, W, _ = prediction.shape

    # Convert Prediction depth to point Map without scaling 
    Z = prediction[..., 2]

    u = torch.arange(W, device=prediction.device).view(1, 1, 1, W).expand(B, K, H, W)
    v = torch.arange(H, device=prediction.device).view(1, 1, H, 1).expand(B, K, H, W)

    
    fx = intrinsics[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    fy = intrinsics[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)

    # Broadcast to shape (B, K, H, W)
    fx = fx.expand(B, K, H, W)
    fy = fy.expand(B, K, H, W)
    cx = cx.expand(B, K, H, W)
    cy = cy.expand(B, K, H, W)

    # Compute X, Y
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    point_map = torch.stack([X, Y, Z], dim=-1)  # (B, K, H, W, 3)

    x_hat = prediction[...,:3]       # Predicted 3D pointmap
    c = prediction[..., 3:4]           # Confidence, keep dimensions [B, K, H, W, 1]

    gt = point_map                      # Ground truth 3D pointmap

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
    # Simulated intrinsics [B, K, 3, 4] (fx, fy = 100, cx, cy = W/2, H/2)
    intrinsics = torch.zeros(B, K, 3, 4)
    intrinsics[:, :, 0, 0] = 1  # fx
    intrinsics[:, :, 1, 1] = 1  # fy
    intrinsics[:, :, 0, 2] = W / 2  # cx
    intrinsics[:, :, 1, 2] = H / 2  # cy
    
    # Ground truth depth
    depth_gt = torch.rand(B, K, H, W, 1) * 5.0  # depth in range [0, 5]

    # Predicted: [x, y, z, confidence]
    pred = torch.rand(B, K, H, W, 4)
    pred[..., 3] = torch.clamp(pred[..., 3], min=0.01)  # avoid log(0)

    # Run the loss
    loss = ConfAlignPointMapRegLoss(depth_gt, pred, intrinsics)
    print("ConfAlignPointMapRegLoss:", loss.item())

    # ----- Test ConfAlignDepthRegLoss -----
    gt_depth = torch.rand(B, K, H, W, 1)
    pred_depth = torch.rand(B, K, H, W, 2)
    pred_depth[..., 1] = torch.clamp(pred_depth[..., 1], min=0.01)  # ensure confidence isn't zero

    loss_depth = ConfAlignDepthRegLoss(gt_depth, pred_depth)
    print("Depth Loss:", loss_depth.item())

test_losses()
