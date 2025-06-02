import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D_norm
import numpy as np
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import sys
import copy
os.chdir('ddpm')

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--split',default="val", type=str, help='split type')
parser.add_argument('--inference_step',default=600, type=int, help='inference step')
parser.add_argument('--weight_path',default='results/model-300.pt', type=str, help='weight path of a trained diffusion model')
args = parser.parse_args()

with open(f'data/mscoco/oscar_split_ViT-B_32_{args.split}.pkl', 'rb') as f:
    print(f'loading data/mscoco/oscar_split_ViT-B_32_{args.split}.pkl')
    test_tokens = pickle.load(f)

save_tokens = copy.deepcopy(test_tokens)

with open('data/mscoco/normalized_text_embed_mean.pkl', 'rb') as f:
    text_mean = pickle.load(f)
    text_mean = text_mean.cuda().float()


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

model.eval()
model = model.cuda()
inference_step = args.inference_step

for idx, key in tqdm(enumerate(test_tokens['captions'].keys()),dynamic_ncols=True, total=len(test_tokens['captions'])):
    text_embeddings = test_tokens['captions'][key]['embed'].cuda().float()
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    text_embeddings -= text_mean
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    text_embeddings = text_embeddings.unsqueeze(1)*5
    
    generated_data = model.q_sample(text_embeddings, torch.tensor([inference_step,], device='cuda'))   
    generated_data = model.ddim_sample_with_img(generated_data, inference_step=inference_step)
    generated_data = generated_data.squeeze().unsqueeze(0)
    generated_data = F.normalize(generated_data, dim=-1)

    save_tokens['captions'][key]['embed']  = generated_data.cpu().detach().half()

output_file_path = f'data/mscoco/modified_oscar_split_ViT-B_32_{args.split}_{inference_step}.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(save_tokens, f)

print(f'modified tokens are saved to {output_file_path}.')



