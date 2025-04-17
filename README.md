### Task delegation 13-04-25
- Scannet Dataloader
- RGB Feature extraction -> DINO Features (B, (32x32), 512)
- Depth features -> Unproject -> PointMap -> ViT Encoder (B, (32x32), 512)
 
- Cross Attention (DINO x PointMap Encoder) ((B, (32x32), 512) x (B, (32x32), 512))

- Linear+DPT Decoder for PointMap and Depth prediction (B, (32x32), 512)

- Training Losses + Distribuited Training Script


loss
pose - initial emendding from first frame. cosnine sim with next frame's embedding

* karan:
* Tanya:
* Yash:
* Sreeharsha:
