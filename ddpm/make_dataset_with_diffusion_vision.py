import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D_norm
import numpy as np
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--split',default="test", type=str, help='split type')
parser.add_argument('--inference_step',default=600, type=int, help='inference step')
parser.add_argument('--weight_path',default="ddpm/results/model-300.pt", type=str, help='model path')
args = parser.parse_args()

with open(f'/mnt/MONG/C3/image_captioning/data/mscoco/oscar_split_ViT-B_32_{args.split}.pkl', 'rb') as f:
    test_tokens = pickle.load(f)

with open('/mnt/MONG/C3/image_captioning/data/mscoco/normalized_image_embed_mean.pkl', 'rb') as f:
    image_mean = pickle.load(f)
    image_mean = image_mean.cuda()

model_path = args.weight_path


model = GaussianDiffusion1D_norm(
    model=Unet1D(
        dim=512,
        init_dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1
    ),
    seq_length=512,
    timesteps=1000,
    objective='pred_x0',
    sampling_timesteps=5
)

weights = torch.load(model_path)
msg = model.load_state_dict(weights['model'], strict=True)
print(msg)
model.cuda()
model.eval()
inference_step = args.inference_step

for idx in tqdm(test_tokens['images'].keys(),dynamic_ncols=True):
    vision_embeddings = test_tokens['images'][idx]['embed'].cuda()
    vision_embeddings = F.normalize(vision_embeddings)
    vision_embeddings -= image_mean
    vision_embeddings = F.normalize(vision_embeddings)*5
    vision_embeddings = vision_embeddings.unsqueeze(1).to(torch.float32)
    
    generated_data = vision_embeddings.clone()

    with torch.no_grad():
        generated_data = model.ddim_sample_with_img(generated_data, inference_step=inference_step)

    generated_data = generated_data.squeeze(1)
    generated_data = F.normalize(generated_data)
    test_tokens['images'][idx]['embed']  = generated_data.cpu().half()

output_file_path = f'image_captioning/data/mscoco/modified_oscar_split_ViT-B_32_{args.split}.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(test_tokens, f)

print(f'modified tokens are saved to {output_file_path}.')