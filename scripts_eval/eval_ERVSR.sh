#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=3 python -B run.py \
    --mode ERVSR \
    --config config_ERVSR\
    --data RealMCVSR \
    --ckpt_abs_name /mnt4/CS570_term/RefVSR/RefVSR_CVPR2022/ERVSR/checkpoint/train/epoch/ckpt/ERVSR_00100.pytorch \
    --data_offset ../../RefVSR/dataset
    --output_offset ./result
    --quantitative_only
