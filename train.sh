# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

python -m torch.distributed.run --nproc_per_node=2 training.py \
    --run_name pointmap_predict \
    --num_epochs 1000 \
    --batch_size 128 \
    --embed_dim 1024 \
    --num_patches 16 \
    --lr 5e-04 \
    --seed 9 \
    --device_ids 0 1 2\
    --log_dir /data/kmirakho/git/Mov3r-L3D-project/logs/ \
    --model_dir /data/kmirakho/git/Mov3r-L3D-project/models/ \
    --eval_model /data/kmirakho/git/Mov3r-L3D-project/models/model_20.pth \
    --dataset_path /data/kmirakho/git/Mov3r-L3D-project/data/ \
    # --run_eval \
    # --load_model \