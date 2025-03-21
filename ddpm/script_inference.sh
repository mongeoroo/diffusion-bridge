#Diffusion augmentation
CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_text.py --split train --path ddpm/results/model-300.pt --inference_step 600

#Diffusion Bridging validation and text set of vision embeddings
CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_vision.py --split val --path ddpm/results/model-300.pt --inference_step 600

CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_vision.py --split test --path ddpm/results/model-300.pt --inference_step 600

