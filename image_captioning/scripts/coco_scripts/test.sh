# PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:path/to/image_captioning

# 현재 디렉토리를 프로젝트 루트로 변경
cd path/to/image_captioning

CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --config configs/coco.yaml \
    --normalize_prefix \
    --test \
    --checkpoint path/to/weight.ckpt



