# Prepare data

Note: every preprocessed or preextracted step can follow the introduction bellow. We take UCM-Captions dataset as an example, while Sydney-Captions, RSICD, or other datasets share the same operation.

## UCM-Captions

### 1-Download UCM-Captions captions and preprocess them

Download UCM-Captions dataset and copy `dataset.json` file in `data/`.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset.json --output_json data/ucm_cocotalk.json --output_h5 data/ucm_cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Prepare the reference file:
```bash
$ python scripts/prepro_reference_json.py --input_json path/dataset.json --output_json path/dataset_eval_reference.json
```
`prepro_reference_json.py` is to generate the coco-like annotation file for evaluation using coco-caption. Remember to modify the path of `dataset_eval_reference.json` in `captioning/utils/eval_utils.py`




### 2-Resnet features (class-level features)


Download pretrained resnet models. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.

`prepro_feats.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)



### 3-CLIP features (sentence-level features)

For fine-tuning CLIP with remote sensing images, please refers to the  [official OpenAI CLIP repo](https://github.com/openai/CLIP) for more pretraining detail.

For extracting image features from CLIP, please refers to the repo of [jianjieluo team] (https://github.com/jianjieluo/OpenAI-CLIP-Feature) for more extraction detail.