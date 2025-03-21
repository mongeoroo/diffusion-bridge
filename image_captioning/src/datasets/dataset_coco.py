# Code adapted from https://github.com/rmokady/CLIP_prefix_caption/blob/main/train.py

import os
import pickle
import random
import sys
from typing import Tuple

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer

project_root = "/mnt/MONG/C3/ddpm"
sys.path.append(project_root)
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

class ClipCocoDataset(pl.LightningDataModule):
    def __init__(self, cfg, split="train"):
        print("=" * 80)
        print("Data split: ", split)
        print("=" * 80)

        self.split = split

        self.cfg = cfg
        self.remove_mean = self.cfg.data.remove_mean
        self.add_gaussian_noise = self.cfg.data.add_gaussian_noise
        self.pre_add_gaussian_noise = self.cfg.data.pre_add_gaussian_noise
        data_path = self.get_data_path(cfg, split)
        self.prefix_length = cfg.model.prefix_length
        self.normalize_prefix = cfg.model.normalize_prefix
        self.re_normalize_prefix = cfg.model.re_normalize_prefix
        self.use_diffmapper = cfg.data.use_diffmapper
        if self.use_diffmapper:
            self.Diffmapper = GaussianDiffusion1D(
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
            weights = torch.load(self.cfg.diffusion.model_path)
            # 가중치 불러오기
            self.Diffmapper.load_state_dict(weights['model'], strict=True)
            self.Diffmapper.eval()
        ###################
        print("=> Loading all_data pkl")
        with open(data_path, "rb") as f:
            all_data = pickle.load(f)
        print("Number of images is %0d" % len(all_data["images"]))
        print("Number of captions is %0d" % len(all_data["captions"]))
        sys.stdout.flush()

        # {image_id: {"img_path": ..., "embed": ...}}
        self.images = all_data["images"]
        # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
        self.captions = all_data["captions"]

        ###################

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl") and split == 'train':
            print("=> Loading caption_id_2_image_id, captions_tokens, all_len dicts")
            with open(f"{data_path[:-4]}_tokens.pkl", "rb") as f:
                (
                    self.captions_tokens,
                    self.caption_id_2_image_id,
                    self.image_id_2_caption_ids,
                    all_len,
                ) = pickle.load(f)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.decoder.model)
            # {caption_id: image_id}
            print("=> Saving caption_id_2_image_id dict")
            self.caption_id_2_image_id = {
                sentid: self.captions[sentid]["img_id"] for sentid in self.captions
            }

            # {image_id: [caption_id]}
            print("=> Saving image_id_2_caption_id dict")
            self.image_id_2_caption_ids = {}
            for sentid in self.captions:
                image_id = self.caption_id_2_image_id[sentid]
                if image_id not in self.image_id_2_caption_ids:
                    self.image_id_2_caption_ids[image_id] = []
                self.image_id_2_caption_ids[image_id].append(sentid)

            # {caption_id: tokenizer(caption)}
            print("=> Saving captions_tokens dict")
            self.captions_tokens = {
                sentid: torch.tensor(
                    self.tokenizer.encode(self.captions[sentid]["caption"]),
                    dtype=torch.int64,
                )
                for sentid in self.captions
            }
            print("=> Saving all_len dict")
            all_len = torch.tensor(
                [
                    self.captions_tokens[sentid].shape[0]
                    for sentid in self.captions_tokens
                ]
            ).float()

            with open(f"{data_path[:-4]}_tokens.pkl", "wb") as f:
                pickle.dump(
                    [
                        self.captions_tokens,
                        self.caption_id_2_image_id,
                        self.image_id_2_caption_ids,
                        all_len,
                    ],
                    f,
                )

        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max())
        )

        self.output_modality = self.cfg.decoder.modality

        # In testing, input modality must be opposite of output modality to evaluate cross-modal task
        if self.cfg.cross_modal_val:
            self.condition = self.split != "train"
        else:
            self.condition = self.split == "test"
        
        # self.condition = self.cfg.condition

        # Get all caption and image ids
        self.img_ids = sorted(list(self.images.keys()))
        random.shuffle(self.img_ids)
        self.cap_ids = sorted(list(self.captions.keys()))
        random.shuffle(self.cap_ids)

        # Sample data
        if "train" in self.split and not OmegaConf.is_none(cfg.data, "sample_frac"):
            img_sample_size = int(len(self.img_ids) * cfg.data.sample_frac)
            cap_sample_size = int(len(self.cap_ids) * cfg.data.sample_frac)
            self.img_ids = random.sample(self.img_ids, img_sample_size)
            self.cap_ids = random.sample(self.cap_ids, cap_sample_size)

        if self.split == "val" and self.cfg.cross_modal_val:
            # Downsample validation set because running generation
            print("=> Subsample 1k examples from validation set for generation")
            img_sample_size = 1000
            self.img_ids = random.sample(self.img_ids, img_sample_size)

        # Load means gap
        with open(cfg.data.text_embed_mean_path, "rb") as f:
            self.text_embed_mean = pickle.load(f)

        with open(cfg.data.image_embed_mean_path, "rb") as f:
            self.image_embed_mean = pickle.load(f)

        self.std = cfg.noise_level

    def get_data_path(self, cfg, split):
        if split == "train":
            data_path = cfg.data.train_data_path
        elif split == "val":
            data_path = cfg.data.val_data_path
        elif split == "test":
            data_path = cfg.data.test_data_path
        else:
            raise NotImplementedError(f"split {split} invalid")

        return data_path

    def __len__(self) -> int:
        if self.condition:
            # Image captioning testing
            return len(self.img_ids)
        else:
            return len(self.cap_ids)

    def pad_tokens(self, item: int):
        """
        Note: this is only for language generation
        (the image padding is in the forward fn of ViT Decoder)
        """
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(self.prefix_length), mask), dim=0
        )  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # For testing, assume cross-modal

        if self.condition:
            return self.get_item_per_image(item)

        item = self.cap_ids[item]
        img_id = self.caption_id_2_image_id[item]
        caption = self.captions[item]["caption"]

        text_prefix = self.captions[item]["embed"].float().squeeze()

        if self.normalize_prefix:
            text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)

        if self.remove_mean:
            text_prefix -= self.text_embed_mean.squeeze()
        
        if self.pre_add_gaussian_noise:
            text_prefix += torch.randn(text_prefix.shape) * self.std
        
        if self.use_diffmapper:
            text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)
            text_prefix = text_prefix.squeeze().unsqueeze(0).unsqueeze(1)
            text_prefix = text_prefix*20
            if not self.pre_add_gaussian_noise:
                text_prefix = self.Diffmapper.q_sample(text_prefix, torch.tensor([self.cfg.diffusion.inference_step,]))
            with torch.no_grad():
                text_prefix = self.Diffmapper.ddim_sample_with_img(text_prefix, inference_step=self.cfg.diffusion.inference_step)
            text_prefix = text_prefix.squeeze()/20        

        if self.add_gaussian_noise:
            text_prefix += torch.randn(text_prefix.shape) * self.std

        tokens, mask = self.pad_tokens(item)
        label = (tokens, mask)

        # Re-normalize
        if self.re_normalize_prefix:
            text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)

        return text_prefix, label, caption, img_id, item

    def get_item_per_image(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for iterating over images (image captioning)
        img_id = self.img_ids[item]
        img_prefix = self.images[img_id]["embed"].float().squeeze()

        if self.normalize_prefix:
            img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)

        if self.remove_mean:
            img_prefix -= self.image_embed_mean.squeeze()

        if self.use_diffmapper:
            img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)
            img_prefix = img_prefix.squeeze().unsqueeze(0).unsqueeze(1)
            img_prefix = img_prefix*20
            with torch.no_grad():
                img_prefix = self.Diffmapper.ddim_sample_with_img(img_prefix, inference_step=self.cfg.diffusion.inference_step)
            img_prefix = img_prefix.squeeze()/20

        # dummy_prefix = torch.zeros_like(img_prefix)
        dummy_tokens = torch.zeros(self.max_seq_len)
        dummy_mask = torch.cat((torch.ones(self.prefix_length), dummy_tokens), dim=0)

        caption_ids = self.image_id_2_caption_ids[img_id]
        captions = [self.captions[c]["caption"] for c in caption_ids]
        
        # captions_embeds = [self.captions[c]["embed"] for c in caption_ids]
        # captions_embeds = torch.stack(captions_embeds)
        # captions_embeds = torch.nn.functional.normalize(captions_embeds, dim=-1)
        # captions_embeds -= self.text_embed_mean
        # captions_embeds = torch.nn.functional.normalize(captions_embeds, dim=-1)
        # captions_embeds = captions_embeds.squeeze(1)
        # img_embeds = img_prefix.half().unsqueeze(0)
        # print(img_embeds@captions_embeds.T)
        # return captions_embeds[0], (dummy_tokens, dummy_mask), captions, img_id, item
        # Re-normalize
        if self.re_normalize_prefix:
            img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)
        return img_prefix.half(), (dummy_tokens, dummy_mask), captions, img_id, item


## To get stuff:
# image_path = self.images[img_id]["img_path"]
# image_embed = self.images[img_id]["embed"]
# caption = self.captions[sent_id]["caption"]
# image_id_for_caption = self.captions[sent_id]["img_id"]
# caption_embed = self.captions[sent_id]["embed"]