# MVP——Multi-View feature fusion and Prompt-based model

## Welcome!
This repository contains the PyTorch implementation of our MVP model in the paper: "**Multi-View Feature Fusion and Visual Prompt for Remote Sensing Image Captioning**".

For more information, please see our early access paper in [IEEE](https://ieeexplore.ieee.org/document/10609353) (Accepted by TGRS 2024)

## Requirements
- Python 3.8
- PyTorch 1.8.0+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)

## Pretrained models
The pretrained CLIP(ViT-B/16 version) model can be downloaded from [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).

## Prepare data
See details in data/README.md

## Start Training
```bash
sh train.sh
```
`--id` in `train.sh` refers to the model name.
The train script will dump checkpoints into the folder specified by `--checkpoint_path`.
For all the arguments, you can specify them in a yaml file and use `--cfg` to use the configurations in that yaml file.


## Evaluation
```bash
sh test.sh
```

`--id` in `test.sh` also means the model name, which is the same in `train.sh`.  


## Reference

If you find this repo useful, please consider citing:

```
@article{wang2024multi,
  title={Multi-View Feature Fusion and Visual Prompt for Remote Sensing Image Captioning},
  author={Wang, Shuang and Lin, Qiaoling and Ye, Xiutiao and Liao, Yu and Quan, Dou and Jin, Zhongqian and Hou, Biao and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements
Our code is based on the codebase repo of [ruotianluo](https://github.com/ruotianluo/ImageCaptioning.pytorch). If you want to replicate the work, maybe carefully follow the codebase repo first.  

Thank the excellent contributors ruotianluo team and [jianjieluo team](https://github.com/jianjieluo/OpenAI-CLIP-Feature).