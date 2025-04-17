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
    --batch_size 8 \
    --context_length 32\
    --embed_dim 1024 \
    --encoder_dim 512 \
    --decoder_dim 512 \
    --num_heads 8 \
    --num_workers 4 \
    --dropout 0.4 \
    --ca_depth 4 \
    --num_patches 16 \
    --lr 5e-04 \
    --seed 9 \
    --device_ids 3\
    --log_dir /data/kmirakho/git/Mov3r-L3D-project/logs/ \
    --model_dir /data/kmirakho/git/Mov3r-L3D-project/models/ \
    --eval_model /data/kmirakho/git/Mov3r-L3D-project/models/model_20.pth \
    --dataset_path /data/kmirakho/git/Mov3r-L3D-project/data/ \
    # --run_eval \
    # --load_model \