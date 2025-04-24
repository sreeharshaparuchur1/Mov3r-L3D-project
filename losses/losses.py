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

    prediction_pm = prediction[0]
    prediction_pm_c = prediction[1]

    mask_pred_pm = torch.isfinite(prediction_pm) #Prediction contains NaN or Inf values
    mask_pred_pm_c = torch.isfinite(prediction_pm_c) #Prediction contains NaN or Inf values

    assert prediction_pm.shape[-1] == 3
    # assert gt_batch.shape[-1] == 1
    assert intrinsics.shape[-1] == 4
    assert intrinsics.shape[-2] == 4

    B, K, H, W, _ = prediction_pm.shape

    # Convert Prediction depth to point Map without scaling 
    # Z = prediction_pm[:,:,:,:, 2]
    Z = gt_batch[:,:,:,:, 0]  # Use ground truth depth for point map calculation

    u = torch.arange(W, device=prediction_pm.device).view(1, 1, 1, W).expand(B, K, H, W).contiguous()
    v = torch.arange(H, device=prediction_pm.device).view(1, 1, H, 1).expand(B, K, H, W).contiguous()

    intrinsics = intrinsics.view(B, K, 4, 4).contiguous()  # Reshape to [B, K, 4, 4]
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

    gt_point_map = torch.stack([X, Y, Z], dim=-1)  # (B, K, H, W, 3)
    mask_gt = torch.isfinite(gt_point_map)  # Mask for ground truth point map
    
    x_hat = prediction_pm     # Predicted 3D pointmap
    c = mask_pred_pm_c*prediction_pm_c # Confidence, keep dimensions [B, K, H, W, 1]
    c = torch.clamp(c, min=eps)  # Avoid log(0)

    # Assuming scale is 1, alignment term becomes simple L2 diff weighted by confidence
    diff = (x_hat - gt_point_map)* mask_pred_pm * mask_gt   # Apply masks to the loss
    loss_data = c * (diff ** 2)  # Confidence-weighted squared error
    loss_data = loss_data  

    # Confidence regularization: -α log(c)
    conf_reg = -alpha * torch.log(c)

    total_loss = loss_data + conf_reg # [B, K, H, W, 3]

    #assert loss is not NaN
    assert torch.all(torch.isfinite(total_loss)) #Loss contains NaN or Inf values"
    return total_loss.mean()
    # return total_loss

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

    prediction_d = prediction[0]
    prediction_d_c = prediction[1]

    prediction_d_c = torch.clamp(prediction_d_c, min=eps)  # avoid log(0)
    
    
    x_hat = prediction_d       # Predicted Depth
    conf = prediction_d_c  # Confidence, keep dimensions [B, K, H, W, 1]
    mask_x_hat = torch.isfinite(x_hat)  # Mask for predicted depth
    mask_x_hat_c = torch.isfinite(prediction_d_c)  # Mask for predicted confidence

    conf = mask_x_hat_c * prediction_d_c  # Apply mask to confidence
    # conf = torch.clamp(conf, min=eps)  # Avoid log(0)

    gt = gt_batch                      # Ground truth Depth
    mask_gt = torch.isfinite(gt)       # Mask for ground truth depth
    
    # Assuming scale is 1, alignment term becomes simple L2 diff weighted by confidence
    diff = (x_hat - gt)*mask_x_hat*mask_gt                  # [B, K, H, W, 1]
    loss_data = conf * (diff ** 2)       # Confidence-weighted squared error

    # Confidence regularization: -α log(c)
    conf_reg = -alpha * torch.log(conf)

    total_loss = loss_data + conf_reg # [B, K, H, W, 1]
    assert torch.all(torch.isfinite(total_loss)) #Loss contains NaN or Inf values
    return total_loss.mean()

def PoseLoss(gt_pose, predict_pose): #, gt_trans, predict_trans):
        # [B, S, 7]
        # assert -> Any assert for shape check here?
        loss = F.mse_loss(predict_pose, gt_pose, reduction='sum')
        return loss 

def test_losses():
    B, K, H, W = 4, 3, 64, 64

    # ----- Test ConfAlignPointMapRegLoss -----
    # Simulated intrinsics [B, K, 3, 4] (fx, fy = 100, cx, cy = W/2, H/2)
    intrinsics = torch.zeros(B, K, 4, 4)
    intrinsics[:, :, 0, 0] = 1  # fx
    intrinsics[:, :, 1, 1] = 1  # fy
    intrinsics[:, :, 0, 2] = W / 2  # cx
    intrinsics[:, :, 1, 2] = H / 2  # cy
    intrinsics[:, :, 3, 3] = 1
    # Ground truth depth
    depth_gt = torch.rand(B, K, H, W, 1) * 5.0  # depth in range [0, 5]

    # Predicted: [x, y, z, confidence]
    # pred = torch.rand(B, K, H, W, 4)
    # pred[..., 3] = torch.clamp(pred[..., 3], min=0.01)  # avoid log(0)
    
    pred = torch.rand(2, B, K, H, W, 3)
    pred[1, ...] = torch.clamp(pred[1, ...], min=0.01)  # avoid log(0)

    # Run the loss
    loss = ConfAlignPointMapRegLoss(depth_gt, pred, intrinsics)
    print("ConfAlignPointMapRegLoss:", loss.item())

    # ----- Test ConfAlignDepthRegLoss -----
    gt_depth = torch.rand(B, K, H, W, 1)
    pred_depth = torch.rand(B, K, H, W, 2)
    pred_depth[..., 1] = torch.clamp(pred_depth[..., 1], min=0.01)  # ensure confidence isn't zero

    loss_depth = ConfAlignDepthRegLoss(gt_depth, pred_depth)
    print("Depth Loss:", loss_depth.item())

if __name__ == "__main__":
    test_losses()
