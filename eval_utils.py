from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import captioning.utils.misc as utils
from functools import reduce


# load coco-caption if available
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    elif 'ucm' in dataset:
        annFile = "/path/ucm_eval_reference.json"
    elif 'sydney' in dataset:
        annFile = "/path/sydney_eval_reference.json"
    elif 'rsicd' in dataset:
        annFile = "/path/rsicd_eval_reference.json"
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    epoch = eval_kwargs["max_epochs"]
    bs = eval_kwargs["beam_size"]
    eval_oracle = eval_kwargs.get('eval_oracle', 0)
    choice = eval_kwargs.get('choice', "best")

    
    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    if "perplexity" in preds_filt[0].keys():
        mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
        mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, df="corpus")
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    if "perplexity" in preds_filt[0].keys():
        out['perplexity'] = mean_perplexity
        out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider}, outfile)
        
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    if not os.path.isdir('eval_results/' + str(model_id)):
        os.mkdir('eval_results/' + str(model_id))
    if split == "test":
        outfile_path = os.path.join('eval_results/', str(model_id), model_id + '_' + split + '_ep' + str(epoch) + '_bs' + str(bs) + '_' + str(choice) + '.json')
    elif ('rsicd' in dataset) and split == 'val':
        outfile_path = os.path.join('eval_results/', str(model_id), model_id + '_' + split + '_ep' + str(epoch) + '_bs' + str(bs) + '_' + str(choice) + '.json')
    else:
        outfile_path = os.path.join('eval_results/', str(model_id), model_id + '_' + split + '.json')

    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile, indent=4)

    return out

def generate_beam(model, tokenizer, beam_size: int = 3, prompt=None, embed=None,
                  entry_length=25, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.module.model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.module.model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.module.model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts

def subsequent_mask(seq):   # GPT2 att_mask
    "Mask out subsequent positions."
    attn_shape = (seq.size(0), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



def generate2(model, tokenizer, tokens=None, prompt=None, embed=None,
              entry_count=1, entry_length=25,  # maximum number of words
              top_p=0.8, temperature=1., stop_token: str = '.'):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.module.model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.module.model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.module.model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]



def replace_char(s, oldChar, newChar ):
    return reduce(lambda s, char: s.replace(char, newChar), oldChar, s)


def eval_gpt(model, crit, loader, tokenizer, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')
    prefix_length = eval_kwargs.get('prefix_length', 40)
    use_beam_search = eval_kwargs.get('eval_kwargs', True)  # beam search

    # Make sure in the evaluation mode

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = []  # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])
        if num_images >= 0 and n >= num_images:
            break

        # feature fusion
        tmp = [data['fc_feats'], data['att_feats'], data['rsi_feats'], data['labels'], data['masks'],
               data['att_masks'], data['rsi_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, rsi_feats, labels, masks, att_masks, rsi_masks = tmp


        with torch.no_grad():
            out = model(fc_feats, att_feats, rsi_feats, labels[..., :-1], att_masks, rsi_masks)
            logits = out.logits[:, prefix_length - 1: -1]
            gts = torch.LongTensor(data['gts']).to(device)
            loss = crit(logits.reshape(-1, logits.shape[-1]), gts.flatten()).item()


        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        with torch.no_grad():
            # prefix_embed
            memory = model.module.model.encode(model.module.att_embed(att_feats), model.module.att_embed_rsi(rsi_feats), att_masks, rsi_masks)
            enc_out = memory.flatten(1)
            layer1_out = model.module.model.layer1(enc_out)
            prefix_embed = model.module.model.layer2(layer1_out).view(enc_out.shape[0], prefix_length, -1)   # [bs, prefix_length, 768]

        for i in range(len(data["infos"])):
            if use_beam_search:
                if split == "val":
                    generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed[i].unsqueeze(0))[0]
                elif split == "test":
                    generated_text_prefix_list = generate_beam(model, tokenizer, beam_size=beam_size, embed=prefix_embed[i].unsqueeze(0))
                    # print
                    list_num = 0
                    for k in generated_text_prefix_list:
                        print(k)
                        list_num += 1
                        if list_num == len(generated_text_prefix_list):
                            print("-"*20)
                    generated_text_prefix = generated_text_prefix_list[0]    # best choice
            else:
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed[i].unsqueeze(0))
            generated_text_prefix = replace_char(generated_text_prefix, ".", "") + "."
            entry = {'image_id': data['infos'][i]['id'], 'caption': generated_text_prefix}
            predictions.append(entry)
            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (n, ix1, loss))


    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions),
               os.path.join('eval_results/', '.saved_pred_' + eval_kwargs['id'] + '_' + split + '.pth'))

    lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    if split == "val":
        model.train()
    return loss_sum / loss_evals, predictions, lang_stats




def eval_split(model, crit, loader, opt, eval_kwargs={}):  #  opt, tokenizer=None,
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    use_feat_fusion = eval_kwargs.get('use_feat_fusion', opt.use_feat_fusion)



    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])


        if not use_feat_fusion:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.to(device) if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            if labels is not None and verbose_loss:

                # forward the model to get loss
                with torch.no_grad():
                    loss = crit(model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:]).item()
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

            # forward the model to also get generated samples for each image
            with torch.no_grad():
                tmp_eval_kwargs = eval_kwargs.copy()
                tmp_eval_kwargs.update({'sample_n': 1})
                seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                seq = seq.data
                entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                            (seq > 0).to(seq_logprobs).sum(1) + 1)
                perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                            (seq > 0).to(seq_logprobs).sum(1) + 1)

        else:   # feature fusion
            tmp = [data['fc_feats'], data['att_feats'], data['rsi_feats'], data['labels'], data['masks'],
                   data['att_masks'], data['rsi_masks']]
            tmp = [_.to(device) if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, rsi_feats, labels, masks, att_masks, rsi_masks = tmp


            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, rsi_feats, labels[..., :-1], att_masks, rsi_masks), labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1


            # forward the model to also get generated samples for each image
            with torch.no_grad():

                tmp_eval_kwargs = eval_kwargs.copy()
                tmp_eval_kwargs.update({'sample_n': 1})

                seq, seq_logprobs = model(fc_feats=fc_feats, att_feats=att_feats, rsi_feats=rsi_feats, att_masks=att_masks, rsi_masks=rsi_masks, opt=tmp_eval_kwargs, mode='sample')

                seq = seq.data
                entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)
                perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                if "gpt2" in opt.input_json:
                    print("\n".join([replace_char(replace_char(tokenizer.decode(_["seq"]), "!", ""), ".", "")+"." for _ in model.done_beams[i]]))
                    print('--' * 10)
                else:
                    print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                    print('--' * 10)


        sents = utils.decode_sequence(model.vocab, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)


        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))
    # if lang_eval == 1:
    lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    if split == "val":
        model.train()
    return loss_sum/loss_evals, predictions, lang_stats


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1}) # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / ((_seq>0).to(_sampleLogprobs).sum(1)+1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent, 'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update({'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1}) # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))