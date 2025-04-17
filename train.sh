# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 training.py \
    --run_name pointmap_predict \
    --num_epochs 1000 \
    --batch_size 4 \
    --context_length 16\
    --embed_dim 768 \
    --encoder_dim 768 \
    --decoder_dim 768 \
    --num_heads 8 \
    --num_workers 4 \
    --dropout 0.4 \
    --ca_depth 4 \
    --patch_size 16 \
    --lr 5e-4 \
    --gamma 0.99999\
    --seed 9 \
    --device_ids 0\
    --dino_encoder /data/kmirakho/l3d_proj/Mov3r-L3D-project/pretrained_weights/dinov2_vitb14_reg4_pretrain.pth \
    --log_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/logs/ \
    --model_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/ \
    --eval_model /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/model_20.pth \
    --dataset_path /data/kmirakho/l3d_proj/scannetv2 \
    # --run_eval \
    # --load_model \