# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_harsha.py\
    --run_name eval_embed_v3 \
    --max_scenes 1 \
    --num_epochs 1 \
    --batch_size 1 \
    --context_length 1 \
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
    --clip_grad_norm 5.0 \
    --clip_grad_val 10.0 \
    --lr 1e-4 \
    --gamma 0.995 \
    --weight_decay 1e-5 \
    --pc_dec_depth 8 \
    --alpha_pointmap 0.5 \
    --alpha_depth 0.5 \
    --eps 1e-6 \
    --seed 9 \
    --save_after 10 \
    --depth_embedder ./pretrained_weights/align3r_depthanything.pth \
    --dino_encoder ./pretrained_weights/dinov2_vitb14_reg4_pretrain.pth \
    --log_dir ./logs/ \
    --model_dir ./models/ \
    --ckpt_dir ./ckpt/ \
    --dataset_path /data/kmirakho/l3d_proj/scannetv4 \
    --load_from_ckpt ./ckpt/train_embed_v6/checkpoint-250.pth \
    --save_evals results/scene1 \
    --visualize_results \
    --run_eval \