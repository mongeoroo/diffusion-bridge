import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D_norm, Trainer1D, Dataset1D
import pickle
import numpy as np
import torch.nn.functional as F
import os
os.chdir('path/to/ddpm')

# 데이터셋 로드 함수 추가
def load_dataset(scale=5.0):
    data_path = 'data/coco/oscar_split_ViT-B_32_train.pkl'
    print(f"loading data from {data_path}...")
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)
    with open('data/coco/normalized_text_embed_mean.pkl', 'rb') as f:
        text_mean = pickle.load(f)
    
    captions = all_data["captions"]

    caption_embeddings = [F.normalize(F.normalize(captions[cap_id]["embed"]) - text_mean) for cap_id in captions]
    return scale*torch.tensor(np.array(caption_embeddings)).squeeze().type(torch.float32)

# 데이터셋 로드
training_seq = load_dataset(scale=5)

# 트레이너 설정
dataset = Dataset1D(training_seq.unsqueeze(1))  # 채널 차원을 추가

# 모델 설정
model = Unet1D(
    dim = 512,
    init_dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels = 1  # 입력 벡터의 채널 수를 1로 설정
)

# 확산 모델 설정
diffusion = GaussianDiffusion1D_norm(
    model,
    seq_length = 512,  # 입력 벡터의 길이에 맞게 설정
    timesteps = 1000,
    objective = 'pred_x0'
    )

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 64,
    train_lr = 8e-5,
    train_num_steps = 3000000,         # 총 학습 스텝
    gradient_accumulate_every = 1,    # 그래디언트 누적 스텝
    ema_decay = 0.995,                # 지수 이동 평균 감쇠
    amp = True,                        # 혼합 정밀도 사용
    results_folder = './results'
)

trainer.train()