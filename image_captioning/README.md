# Image Captioning

## Datasets
Download the MSCOCO karpathy-splits caption dataset from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). 

## Getting Started

### Prepare Environment
Create conda environment
```
conda env create -f environment.yml
ln -s [pathtoCOCOdataset] ./data/mscoco
```

Note: if using Imagebind, follow the [official repo](https://github.com/facebookresearch/ImageBind) to create a separate imagebind conda environment.

### Prepare Labels and Embeddings
1. Preprocess COCO labels
```
python3 src/parse_data/create_labels_json.py
```

2. Embed COCO dataset with CLIP and compute modality means
```
python3 src/parse_data/parse_coco.py
python3 src/parse_data/compute_embed_means.py
```

3. (Optional) Embed COCO dataset with ImageBind and compute modality means
```
conda activate imagebind
python3 src/parse_data/parse_coco_imagebind.py
python3 src/parse_data/compute_embed_means_imagebind.py
conda deactivate imagebind
```

## Training

Training, model, logging and data configurations are provided in `configs`. 

Scripts to run experiments on COCO using CLIP are provided in `scripts`.

To train,
```
bash ./scripts/coco_scripts/train.sh
```

To eval,
```
bash ./scripts/coco_scripts/test.sh
```
