#Diffusion augmentation
CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_text.py --split train --weight_path results/model-300.pt --inference_step 400

#Diffusion Bridging validation and text set of vision embeddings
CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_vision.py --split val --weight_path results/model-300.pt --inference_step 600

CUDA_VISIBLE_DEVICES=0 python ddpm/make_dataset_with_diffusion_vision.py --split test --weight_path results/model-300.pt --inference_step 600

