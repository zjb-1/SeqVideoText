#!/usr/bin/env bash

set -x

python3 -u ../seq_video_text/main.py \
    --dataset_file VideoText \
    --ytvis_path /lustre/datasharing/jbzhang/data/ \
    --train_img_prefix YVT/train_images \
    --train_ann_file YVT/video_train.json \
    --test_img_prefix YVT/test_images \
    --test_ann_file YVT/video_test.json \
    --epochs 50 \
    --checkpoint_step 2 \
    --lr 1e-4 \
    --lr_drop 35 \
    --batch_size 1 \
    --num_workers 2 \
    --num_queries 300 \
    --num_frames 5 \
    --with_box_refine \
    --masks \
    --obj \
    --rel_coord \
    --backbone resnet50 \
    --output_dir r2 \
    --start_epoch 0 \
    --pretrain_weights ../exp_jointvideo/r2_new/checkpoint0007.pth \
    # --rec \

