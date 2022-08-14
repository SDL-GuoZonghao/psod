#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
GPU_NUM=8

# CONFIG='configs/psod/lss_roi_pooling_dual_vit_small_faster_rcnn_fpn_voc0712_1x.py'
# WORK_DIR='../work_dirs/mmpsod/lss_roi_pooling_dual_vit_small_faster_rcnn_fpn_voc0712_1x_scale7'

CONFIG='configs/psod/lss_attn_pooling_dual_vit_small_imted_fpn_voc0712_1x.py'
WORK_DIR='../work_dirs/mmpsod/imted_psod_scale7_attn_pooling_norm_filter0.2_shortcut_mlp_voc0712_1x'


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50041 --use_env ./tools/train.py \
    ${CONFIG} --cfg-options \
        model.backbone.dual_depth=0 \
    --work-dir ${WORK_DIR} \
    --gpus ${GPU_NUM} --launcher pytorch
    

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50041 --use_env ./tools/test.py \
#     ${CONFIG} \
#     ${WORK_DIR}/epoch_1.pth \
#     --eval mAP --launcher pytorch