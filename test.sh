#!/bin/bash
id="ucm_model" # this is same as the model name in `train.sh`

python eval.py --id_name $id --choice best --dump_images 0 --num_images -1 --language_eval 1 --split test --beam_size 5