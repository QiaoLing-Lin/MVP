import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag):
        opt = self.opt
        
        out = {}

        reduction = 'none' if drop_worst_flag else 'mean'
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts, reduction=reduction)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            gts_t = torch.LongTensor(gts)  # tensor
            pad = np.zeros((gts_t.shape[0], gts_t.shape[1], 1), dtype='uint32')  # 补0
            gts_arr = np.array(gts)
            gts_0 = np.concatenate((gts_arr, pad), axis=2)  # cat
            gts = torch.LongTensor(gts_0.tolist()).to(device)   # tensor
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), gts, masks[..., 1:], reduction=reduction)


        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out

# feat fusion
class LossWrapper_(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper_, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            if (self.opt.caption_model == "clipfeatsfusiongpt") or (self.opt.caption_model == "clipffgpt"):
                self.crit = torch.nn.CrossEntropyLoss(ignore_index=0)
            else:
                self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion_(opt)


    def forward(self, fc_feats, att_feats, rsi_feats, labels, masks, att_masks, rsi_masks, gts, gt_indices, sc_flag):
        opt = self.opt

        out = {}
        if not sc_flag:
            if self.opt.caption_model == "clipfeatsfusiongpt":
                gts = torch.LongTensor(gts).cuda()
                out = self.model(fc_feats, att_feats, rsi_feats, labels[..., :-1], att_masks, rsi_masks)  # model output
                logits = out.logits[:, self.opt.prefix_length - 1: -1]  # 选出除去prefix的tokens
                loss = self.crit(logits.reshape(-1, logits.shape[-1]), gts.flatten())

            else:
                # out = self.model(fc_feats, att_feats, rsi_feats, labels[..., :-1], att_masks, rsi_masks)  # [bs*5, 21, vocab.size]
                loss = self.crit(self.model(fc_feats, att_feats, rsi_feats, labels[..., :-1], att_masks, rsi_masks), labels[..., 1:],
                             masks[..., 1:])

        else:
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, rsi_feats, att_masks, rsi_masks,
                                                     opt={'sample_method': opt.train_sample_method,
                                                          'beam_size': opt.train_beam_size,
                                                          'sample_n': opt.train_sample_n},
                                                     mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            output = self.rl_crit(sample_logprobs, gen_result, gts)
            loss = output['loss']
            out['reward'] = output['reward']
        out['loss'] = loss
        return out
