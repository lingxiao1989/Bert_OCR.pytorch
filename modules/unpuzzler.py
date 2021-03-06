# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    '''参数设置'''
    """ Relation Attention Module """
    p_drop_attn = 0.1
    p_drop_hidden = 0.1
    dim = 512                       # the encode output feature
    attention_layers = 2                    # the layers of transformer
    n_heads = 8
    dim_ff = 1024 * 2                       # 位置前向传播的隐含层维度

    ''' Parallel Attention Module '''
    dim_c = dim
    max_vocab_size = 26             # 一张图片含有字符的最大长度

    """ Two-stage Decoder """
    len_alphabet = 39               # 字符类别数量
    decoder_atten_layers = 2

class Puzzle_Solver(nn.Module):
    def __init__(self, opt):
        super(Puzzle_Solver, self).__init__()
        self.opt = opt
        self.stages = {'Trans': "None", 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        cfg = Config()
        cfg.dim = opt.output_channel; cfg.dim_c = opt.output_channel              # 降维减少计算量
        cfg.p_dim = opt.position_dim                        # 一张图片cnn编码之后的特征序列长度
        cfg.max_vocab_size = opt.batch_max_length + 1                # 一张图片中最多的文字个数, +1 for EOS
        cfg.len_alphabet = opt.alphabet_size                # 文字的类别个数
        self.SequenceModeling = Bert_Ocr(cfg)

    def forward(self, input, text, is_train=True):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        pad_mask = text
        contextual_feature = self.SequenceModeling(visual_feature, pad_mask)

class Bert_Ocr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg, cfg.attention_layers)
        #self.attention = Parallel_Attention(cfg)
#         self.attention = MultiHeadAttention(d_model=cfg.dim, max_vocab_size=cfg.max_vocab_size)
        #self.decoder = Two_Stage_Decoder(cfg)

    def forward(self, encoder_feature, mask=None):
        bert_out = self.transformer(encoder_feature, mask)                 # 做一个self_attention//4*200*512
        #glimpses = self.attention(encoder_feature, bert_out, mask)         # 原始序列和目标序列的转化//4*512*94
        #self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        #res = self.decoder(glimpses.transpose(1,2))
        return bert_out

class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    """ consist of: Embeddings, Block"""
    def __init__(self, cfg, n_layers):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, x, mask):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, mask)
        return h

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.pos_embed = nn.Embedding(cfg.p_dim, cfg.dim) # position embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)    # (S,) -> (B, S)

        e = x + self.pos_embed(pos) 
        return self.drop(self.norm(e))

class Block(nn.Module):
    """ Transformer Block """
    """ consist of: MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward"""
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h
        
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)
    
def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

class Two_Stage_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_w = nn.Linear(cfg.dim_c, cfg.len_alphabet)
        self.relation_attention = Transformer(cfg, cfg.decoder_atten_layers)
        self.out_w1 = nn.Linear(cfg.dim_c, cfg.len_alphabet)

    def forward(self, x):
        x1 = self.out_w(x)
        x2 = self.relation_attention(x, mask=None)
        x2 = self.out_w1(x2)                    # 两个分支的输出部分采用不同的网络

        return x1, x2

'''from image transformer:

class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, hparams):
        super().__init__()
        self.attn = Attn(hparams)
        self.hparams = hparams
        self.dropout = nn.Dropout(p=hparams.dropout)
        self.layernorm_attn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.hparams.filter_size, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hparams.filter_size, self.hparams.hidden_size, bias=True))

    def preprocess_(self, X):
        return X

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        X = self.preprocess_(X)
        y = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(self.preprocess_(X))
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X

class Attn(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.kd = self.hparams.total_key_depth or self.hparams.hidden_size
        self.vd = self.hparams.total_value_depth or self.hparams.hidden_size
        self.q_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hparams.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hparams.hidden_size, bias=False)
        assert self.kd % self.hparams.num_heads == 0
        assert self.vd % self.hparams.num_heads == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.hparams.num_heads, self.vd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.hparams.num_heads) ** (-0.5)

        if self.hparams.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.hparams.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.hparams.block_length
            pad = (0, 0, 0, (-len) % self.hparams.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch, nheads, (nblocks - 1), blen * 2, depth]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3)
                tail_q = q[:,:,1:]
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
                result = torch.cat([first_output, tail_output], 2)
                result = result[:,:,:X.shape[1],:]
            else:
                result = first_output[:,:,:X.shape[1],:]

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        return result
'''