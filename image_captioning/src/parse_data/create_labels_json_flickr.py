import json
from tqdm import tqdm
import sys
import os
import math
# 프로젝트의 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = "/mnt/MONG/C3/image_captioning"
MASTER_JSON = f"{DATA_ROOT}/data/flickr30k/dataset_flickr30k.json"
splits = ["train", "val", "test"]

LABELS_JSONS_LST = dict(
    zip(
        splits,
        [f"{DATA_ROOT}/data/flickr30k/labels_{split}.json" for split in splits],
    )
)


def create_labels_json():
    all_labels = dict(
        zip(
            splits,
            [
                {"annotations": [], "images": []},
                {"annotations": [], "images": []},
                {"annotations": [], "images": []},
            ],
        )
    )

    out_paths = LABELS_JSONS_LST

    with open(MASTER_JSON, "r") as f:
        data = json.load(f)["images"]

    for i in tqdm(range(len(data))):
        d = data[i]
        img_id, split = d["imgid"], d["split"]

        if split == "restval":
            split = "train"

        image_dict = {"id": img_id}
        all_labels[split]["images"].append(image_dict)

        for caption_data in d["sentences"]:
            assert img_id == caption_data["imgid"]
            sent_id, caption = caption_data["sentid"], caption_data["raw"]

            caption_dict = {"image_id": img_id, "caption": caption, "id": sent_id}

            all_labels[split]["annotations"].append(caption_dict)

        if (i + 1) % 10000 == 0:
            for split in splits:
                out_path = out_paths[split]
                with open(out_path, "w") as f:
                    json.dump(all_labels[split], f)

                print("Total number of annotations (so far)")
                print_anns_totals(all_labels)

    for split in splits:
        out_path = out_paths[split]
        with open(out_path, "w") as f:
            json.dump(all_labels[split], f)

    print("Total number of annotations")
    print_anns_totals(all_labels)


def print_anns_totals(all_labels):
    anns_totals = [len(all_labels[split]["annotations"]) for split in splits]
    print(dict(zip(splits, anns_totals)))
    print("Total number of images")
    imgs_totals = [len(all_labels[split]["images"]) for split in splits]
    print(dict(zip(splits, imgs_totals)))


if __name__ == "__main__":
    create_labels_json()
