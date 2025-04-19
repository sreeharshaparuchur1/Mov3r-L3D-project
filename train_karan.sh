# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,3 python -m torch.distributed.run --nproc_per_node=3 --master_port=29500 training_karan.py \
    --run_name train_embed_v1 \
    --max_scenes 10\
    --num_epochs 1000 \
    --batch_size 4 \
    --context_length 16\
    --num_heads 8 \
    --img_size 224 \
    --ds_num_workers 8\
    --dl_num_workers 4 \
    --prefetch_factor 8\
    --embed_dim 768 \
    --encoder_dim 768 \
    --decoder_dim 768 \
    --load_after 5\
    --frame_skip 5\
    --ca_depth 4 \
    --patch_size 16 \
    --dropout 0.4 \
    --lr 1e-4 \
    --gamma 0.999\
    --weight_decay 1e-5\
    --pc_dec_depth 8 \
    --seed 9 \
    --save_after 50\
    --dino_encoder /data/kmirakho/l3d_proj/Mov3r-L3D-project/pretrained_weights/dinov2_vitb14_reg4_pretrain.pth \
    --log_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/logs/ \
    --model_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/ \
    --ckpt_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/ckpt/ \
    --eval_model /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/model_20.pth \
    --dataset_path /data/kmirakho/l3d_proj/scannetv2 \
    # --load_from_ckpt /data/kmirakho/l3d_proj/Mov3r-L3D-project/ckpt/train_embed_smallv3/checkpoint-0.pth \
    # --run_eval \
    # --load_model \

# python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 training.py \
#     --run_name train_embedding \
#     --num_epochs 1000 \
#     --batch_size 4 \
#     --context_length 16\
#     --max_scenes 2\
#     --embed_dim 768 \
#     --encoder_dim 768 \
#     --decoder_dim 768 \
#     --num_heads 8 \
#     --ds_num_workers 8\
#     --dl_num_workers 4 \
#     --prefetch_factor 8\
#     --load_after 5\
#     --frame_skip 5\
#     --dropout 0.4 \
#     --ca_depth 4 \
#     --patch_size 16 \
#     --lr 5e-4 \
#     --gamma 0.999\
#     --seed 9 \
#     --save_after 50\
#     --device_ids 1\
#     --dino_encoder /data/kmirakho/l3d_proj/Mov3r-L3D-project/pretrained_weights/dinov2_vitb14_reg4_pretrain.pth \
#     --log_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/logs/ \
#     --model_dir /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/ \
#     --eval_model /data/kmirakho/l3d_proj/Mov3r-L3D-project/models/model_20.pth \
#     --dataset_path /data/kmirakho/l3d_proj/scannetv2 \
#     # --run_eval \
#     # --load_model \