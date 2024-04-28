# Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning

## Setup

```bash
conda create --name troika python=3.7
conda activate troika
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# pip3 install git+https://github.com/openai/CLIP.git
```

Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

### CLIP Download

Due to network issues, our implementation requires manual downloading of the CLIP checkpoint:

```bash
cd <CLIP_MODEL_ROOT>
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

If you want to automatically download the checkpoint through the `clip` package, please modify the relevant code yourself.

## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.

```bash
sh download_data.sh
```

If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`<DATASET_ROOT>/<DATASET>` where `<DATASET> = {'mit-states', 'ut-zappos', 'cgqa'}`.

## Training

```py
python -u train.py \
--clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/troika/<DATASET>.yml \
--num_workers 10 \
--seed 0
```

## Evaluation

We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation

```py
python -u test.py \
--clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/troika/<DATASET>.yml \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```

### Open-World Evaluation

For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.

For feasbility calibration, we have computed feasibility similarities and saved them at `data/feasibility_<dataset>.pt`. Therefore, you don't need to handle this yourself. If you need to compute on your own, please refer to [DFSP](https://github.com/Forest-art/DFSP?tab=readme-ov-file#feasibility-calibration).

Just run:

```py
python -u test.py \
--clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/troika/<DATASET>-ow.yml \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```

## Citation

If you find the code useful in your research, please cite our paper:

```
@inproceedings{Huang2024Troika,
    title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
    author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

## Acknowledgement

Our code references the following projects:

* [DFSP](https://github.com/Forest-art/DFSP)
* [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)