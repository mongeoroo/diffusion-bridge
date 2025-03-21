# PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:path/to/image_captioning

# 현재 디렉토리를 프로젝트 루트로 변경
cd path/to/image_captioning

CUDA_VISIBLE_DEVICES=3 python3 run.py \
    --config configs/coco.yaml \
    --normalize_prefix \
    --val_eval \
    --cross_modal_val \
    --train \
    --test \
    --re_normalize_prefix