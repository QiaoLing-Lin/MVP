#!/bin/bash
# ucm
python train.py --id ucm_model --optim radam --cfg configs/transformer/transformer_ucm.yml --checkpoint_path log_ucm/log_ucm_model --max_epochs 35 --prefix_length 5 --batch_size 32
