
import argparse
import json
import os
import pickle

import clip
import skimage.io as io
import torch
from create_labels_json_flickr import DATA_ROOT, MASTER_JSON
from PIL import Image
from tqdm import tqdm

# captions -- {caption_id: {caption_raw: .., image_id: ..}}
# embeddings -- {image_id: embedding}

splits = ["train", "val", "test"]


def main(clip_model_type: str):
    device = torch.device("cuda:3")
    clip_model_name = clip_model_type.replace("/", "_")

    out_paths = [
        f"{DATA_ROOT}/data/flickr30k/karpathy_split_{clip_model_name}_{split}.pkl"
        for split in splits
    ]
    out_paths = dict(zip(splits, out_paths))

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(MASTER_JSON, "r") as f:
        data = json.load(f)["images"]
    print("%0d images loaded from json " % len(data))


    all_images = dict(zip(splits, [{}, {}, {}]))
    all_captions = dict(zip(splits, [{}, {}, {}]))
    for i in tqdm(range(len(data))):
        d = data[i]
        split, filename = d["split"],  d["filename"]
        if split == "restval":
            split = "train"

        # Get and save image and image embed
        img_id = d["imgid"]
        filename = f"{DATA_ROOT}/data/flickr30k/Images/{filename}"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embed = clip_model.encode_image(image).cpu()

        ## Note: changes for all the keys!
        all_images[split][img_id] = {"img_path": filename, "embed": image_embed}

        # Get + save caption and caption embed
        for caption_data in d["sentences"]:
            assert img_id == caption_data["imgid"]
            try:
                with torch.no_grad():
                    text = clip.tokenize(caption_data["raw"]).to(device)
                    text_embed = clip_model.encode_text(text).cpu()
            except:
                print(f"Error with caption {caption_data['raw']}")
                print('image id', img_id)
                continue

            sent_id = caption_data["sentid"]

            ## Note: changes for all the keys!!
            all_captions[split][sent_id] = {
                "caption": caption_data["raw"],
                "img_id": img_id,
                "embed": text_embed,
            }

        if (i + 1) % 10000 == 0:
            for split in splits:
                with open(out_paths[split], "wb") as f:
                    pickle.dump(
                        {"images": all_images[split], "captions": all_captions[split]},
                        f,
                    )

            print_totals(all_images, all_captions)

    for split in splits:
        with open(out_paths[split], "wb") as f:
            pickle.dump(
                {"images": all_images[split], "captions": all_captions[split]}, f
            )

    print("Done")
    print_totals(all_images, all_captions)
    return 0


def print_totals(all_images, all_captions):
    print("Done")
    print("Total number of images (so far)")
    embed_totals = [len(all_images[split].values()) for split in splits]
    print(dict(zip(splits, embed_totals)))
    print("Total number of captions (so far)")
    caption_totals = [len(all_captions[split].values()) for split in splits]
    print(dict(zip(splits, caption_totals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_type",
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32", "ViT-L/14"),
    )
    args = parser.parse_args()
    exit(main(args.clip_model_type))
