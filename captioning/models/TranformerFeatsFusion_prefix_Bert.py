# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8
###################################  rsi features (iod features) ###################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel_ import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
from transformers import BertModel, BertConfig
import os

class HookTool:
    def __init__(self):
        self.feat = None

    def hook_fun(self, module, feat_in, feat_out):
        self.feat = feat_out



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, opt):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        ##########################
        # prefix mapping net
        ##########################
        self.opt = opt
        self.prefix_feat = self.opt.prefix_feat
        if self.prefix_feat == "rsclip" or self.prefix_feat == "rn" or self.prefix_feat == "clip":
            self.prefix_len = self.opt.prefix_length  # 5
        elif self.prefix_feat == "rsclip+rn":
            self.prefix_len = self.opt.prefix_length*2
        elif self.prefix_feat == "None" and self.opt.use_learnable_prefix == True:
            self.prefix_len = self.opt.prefix_length
        self.embed_size = self.opt.input_encoding_size # 768
        self.bs = opt.batch_size

        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.idx = -1

    def forward(self, src, rsi_feats, tgt, src_mask, rsi_masks, tgt_mask, prefix_clip, prefix_rn):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, rsi_feats, src_mask, rsi_masks)
        if self.prefix_feat == "rsclip":
            return self.decode(memory, src_mask, tgt, tgt_mask, prefix_clip)  # add prefix feat = rsclip
        elif self.prefix_feat == "rn":
            return self.decode(memory, src_mask, tgt, tgt_mask, prefix_rn)  # add prefix feat = rn101
        elif self.prefix_feat == "rsclip+rn":
            prefix_feat = torch.cat((prefix_clip, prefix_rn), dim=1)
            return self.decode(memory, src_mask, tgt, tgt_mask, prefix_feat)
        else:
            return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, rsi_feats, src_mask, rsi_masks):  # two kind of feats
        return self.encoder(self.src_embed(src), rsi_feats, src_mask, rsi_masks)


    def decode(self, memory, src_mask, tgt, tgt_mask, prefix_feat=None):  # memory = encoder output

        tgt_embeddings = self.decoder.embeddings(tgt)  # bert.embedding ==> [bs*5, 21, 768]
        # visual info mapping to prefix
        if self.opt.use_prefix_feat:
            prefix_feat = utils.repeat_tensors(int(tgt.shape[0]/prefix_feat.shape[0]), prefix_feat)
            prefix = prefix_feat
        else:  # memory adaptive pool
            prefix_feat = torch.nn.functional.adaptive_avg_pool2d(memory, [self.prefix_len, self.embed_size])
            prefix_feat = utils.repeat_tensors(int(tgt.shape[0] / prefix_feat.shape[0]), prefix_feat)
            prefix = prefix_feat
        prefix_tgt_embed = torch.cat((prefix, tgt_embeddings), dim=1)   # [bs*5, prefix_len+21, 768]

        # bert
        prefix_ys_mask = subsequent_mask(self.prefix_len + tgt.size(1)).to(memory.device)

        return self.decoder(inputs_embeds=prefix_tgt_embed,
                                attention_mask=prefix_ys_mask,
                                encoder_hidden_states=memory,
                                encoder_attention_mask=src_mask)[0][:, self.prefix_len:]


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, y, mask, rsi_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, y, mask, rsi_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = LayerNorm(size)  # new
        self.src_attn = src_attn     # src_attn = self.attn

    def forward(self, x, y, mask, rsi_mask):
        "Follow Figure 1 (left) for connections."
        y = self.norm(y)  # y is rsi feature
        x = self.sublayer[0](x, lambda x: self.src_attn(x, y, y, rsi_mask) + self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N, opt):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        # self.CI = Causual_intervention(opt)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # x = x + self.CI(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, opt):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)


    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # text self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # cross-attention
        return self.sublayer[2](x, self.feed_forward)                        # ff


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CLIPFeatsFusionPrefixBERT(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,   # glove_dim=300,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, opt=None):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # Bert as decoder
        dec_config = BertConfig(vocab_size=tgt_vocab,
                                hidden_size=d_model, # 768
                                num_hidden_layers=N_dec, # 6
                                num_attention_heads=h, # 8
                                intermediate_size=d_ff, # 2048
                                hidden_dropout_prob=dropout,
                                attention_probs_dropout_prob=dropout,
                                max_position_embeddings=512,
                                type_vocab_size=1,
                                is_decoder=True,
                                add_cross_attention=True)

        decoder = BertModel.from_pretrained(pretrained_model_name_or_path="/data1/lql/bert-base-uncased", config=dec_config, ignore_mismatched_sizes=True)  # /data2/lql/bert-base-uncased  opt.bert_premodel


        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_enc),
            decoder,
            lambda x: x,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),  # tgt_embed
            Generator(d_model, tgt_vocab),
            opt
            )


        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(CLIPFeatsFusionPrefixBERT, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))

        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        # self.glove_dim = getattr(opt, 'glove_dim', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', opt.num_att_heads)
        self.dropout = getattr(opt, 'dropout', opt.dropout)
        self.rsi_feat_size = getattr(opt, "rsi_feat_size", opt.rsi_feat_size)

        delattr(self, 'att_embed')
        # notice att_feat_size
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        self.att_embed_rsi = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.rsi_feat_size, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     # glove_dim= self.glove_dim,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout,
                                     opt =self.opt)

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, rsi_feats, att_masks, rsi_masks):

        att_feats, rsi_feats, seq, att_masks, rsi_masks, seq_mask = self._prepare_feature_forward(att_feats, rsi_feats, att_masks, rsi_masks)
        memory = self.model.encode(att_feats, rsi_feats, att_masks, rsi_masks)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, rsi_feats, att_masks=None, rsi_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        rsi_feats, rsi_masks = self.clip_att(rsi_feats, rsi_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        rsi_feats = pack_wrapper(self.att_embed_rsi, rsi_feats, rsi_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        rsi_masks = rsi_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:, 0] = 1  # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks, rsi_feats, rsi_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks, rsi_feats, rsi_masks]
                                                            )
        else:
            seq_mask = None

        return att_feats, rsi_feats, seq, att_masks, rsi_masks, seq_mask

    def adapool(self, feats):
        if feats.shape[2] == 768:
            prefix_feat = torch.nn.functional.adaptive_avg_pool2d(feats, [self.opt.prefix_length, 768])
        if feats.shape[2] == 2048:
            prefix_feat = torch.nn.functional.adaptive_avg_pool2d(feats, [self.opt.prefix_length, 768])
        return prefix_feat

    def _forward(self, fc_feats, att_feats, rsi_feats, seq, att_masks=None, rsi_masks=None):
        prefix_clip = self.adapool(att_feats)  # [bs, 196, 768]
        prefix_rn = self.adapool(rsi_feats)    # [bs, 196, 2048]
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, rsi_feats, seq, att_masks, rsi_masks, seq_mask = self._prepare_feature_forward(att_feats, rsi_feats, att_masks, rsi_masks, seq)

        out = self.model(att_feats, rsi_feats, seq, att_masks, rsi_masks, seq_mask, prefix_clip, prefix_rn)

        outputs = self.model.generator(out)

        return outputs


    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, prefix_feats=None):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(memory, mask,
                                ys,
                                subsequent_mask(ys.size(1))
                                .to(memory.device),
                                prefix_feats)  # add prefix feat

        return out[:, -1], [ys.unsqueeze(0)]

