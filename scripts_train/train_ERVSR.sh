#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9002 run.py \
                        --is_train \
                        --mode ERVSR \
                        --config config_ERVSR \
                        --data RealMCVSR \
                        --data_offset ../../RefVSR/dataset
                        -b 8 \
                        -th 2 \
                        -dl \
                        -ss \
                        -dist \
                        --is_crop_valid \

# CUDA_VISIBLE_DEVICES=0 python run.py \
#                         --is_train \
#                         --mode ERVSR \
#                         --config config_ERVSR \
#                         --data RealMCVSR \
#                         --data_offset ../../RefVSR/dataset
#                         -b 1 \
#                         -ss \
#                         --is_crop_valid \
