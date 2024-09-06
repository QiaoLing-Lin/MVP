from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import sys

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader_ff import *   # feature fusion
from captioning.data.dataloaderraw import *
# import captioning.utils.eval_utils as eval_utils
import eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch
import time

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                help='cpu or cuda')
# 方便测试
parser.add_argument('--id_name', type=str, default='', help='an id identifying this run/job.')
parser.add_argument("--choice", type=str, default="best", help="choose test the last or the best weight model")
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

opt.id_name = opt.id_name[:-1]
# Load infos
# log_path = "log_mask"
ds = ['ucm', 'sydney', 'rsicd', 'rsitmd']
# mask
for i in ds:
    if i in opt.id_name:
        if opt.choice == "best":
            opt.model = "log_%s/log_%s/model-best.pth" %(i, opt.id_name)
            opt.infos_path = "log_%s/log_%s/infos_%s-best.pkl" %(i, opt.id_name, opt.id_name)
        elif opt.choice == "last":
            opt.model = "log_%s/log_%s/model.pth" % (i, opt.id_name)
            opt.infos_path = "log_%s/log_%s/infos_%s.pkl" % (i, opt.id_name, opt.id_name)



with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

print(opt)

vocab = infos['vocab'] # ix -> word mapping

# pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
# result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

opt.vocab = vocab

model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model, map_location='cpu'))
# model.load_state_dict(torch.load(opt.model))
model.to(opt.device)
model.eval()
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, opt, vars(opt))
print('loss: ', loss)

if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
