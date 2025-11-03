import os
import json
import copy
import math
from collections import OrderedDict
from typing import Optional, List

import torch
import torch.nn as nn
from numba import jit, prange
import numpy as np
import torch.nn.functional as F

from comp_trans_tts.utils.tools import (
    get_variance_level,
    get_phoneme_level_pitch,
    get_phoneme_level_energy,
    get_mask_from_lengths,
    pad_1D,
    pad,
    dur_to_mel2ph,
)
from comp_trans_tts.utils.pitch_tools import f0_to_coarse, denorm_f0, cwt2f0_norm
from comp_trans_tts.model.transformers.blocks import (
    Embedding,
    SinusoidalPositionalEmbedding,
    LayerNorm,
    LinearNorm,
    ConvNorm,
    ConvBlock,
    ConvBlock2D,
)
from comp_trans_tts.model.transformers.transformer import ScaledDotProductAttention
from comp_trans_tts.model.coordconv import CoordConv2d


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x


class ProsodyExtractor(nn.Module):
    """ Prosody Extractor """

    def __init__(self, n_mel_channels, d_model, kernel_size):
        super(ProsodyExtractor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.Sequential(
            ConvBlock2D(
                in_channels=1,
                out_channels=self.d_model,
                kernel_size=kernel_size,
            ),
            ConvBlock2D(
                in_channels=self.d_model,
                out_channels=1,
                kernel_size=kernel_size,
            ),
        )
        self.gru = nn.GRU(
            input_size=n_mel_channels,
            hidden_size=self.d_model,
            batch_first=True,
            bidirectional=True,
        )

    def get_prosody_embedding(self, mel):
        """
        mel -- [B, mel_len, n_mel_channels], B=1
        h_n -- [B, 2 * d_model], B=1
        """
        x = self.conv_stack(mel.unsqueeze(-1)).squeeze(-1)
        _, h_n = self.gru(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)
        return h_n

    def forward(self, mel, mel_len, duration, src_len):
        """
        mel -- [B, mel_len, n_mel_channels]
        mel_len -- [B,]
        duration -- [B, src_len]
        src_len -- [B,]
        batch -- [B, src_len, 2 * d_model]
        """
        batch = []
        for m, m_l, d, s_l in zip(mel, mel_len, duration, src_len):
            b = []
            for m_p in torch.split(m[:m_l], list(d[:s_l].int()), dim=0):
                b.append(self.get_prosody_embedding(m_p.unsqueeze(0)).squeeze(0))
            batch.append(torch.stack(b, dim=0))

        return pad(batch)


class MDN(nn.Module):
    """ Mixture Density Network """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.w = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, x):
        """
        x -- [B, src_len, in_features]
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        """
        B, src_len, _ = x.shape
        w = self.w(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(B, src_len, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(B, src_len, self.num_gaussians, self.out_features)
        return w, sigma, mu


class ProsodyPredictor(nn.Module):
    """ Prosody Predictor """

    def __init__(self, d_model, kernel_size, num_gaussians, dropout):
        super(ProsodyPredictor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=kernel_size[i],
                    dropout=dropout,
                    normalization=nn.LayerNorm,
                    transpose=True,
                )
                for i in range(2)
            ]
        )
        self.gru_cell = nn.GRUCell(
            self.d_model + 2 * self.d_model,
            2 * self.d_model,
        )
        self.gmm_mdn = MDN(
            in_features=2 * self.d_model,
            out_features=2 * self.d_model,
            num_gaussians=num_gaussians,
        )

    def init_state(self, x):
        """
        x -- [B, src_len, d_model]
        p_0 -- [B, 2 * d_model]
        self.gru_hidden -- [B, 2 * d_model]
        """
        B, _, d_model = x.shape
        p_0 = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        self.gru_hidden = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        return p_0

    def forward(self, h_text, mask=None):
        """
        h_text -- [B, src_len, d_model]
        mask -- [B, src_len]
        outputs -- [B, src_len, 2 * d_model]
        """
        x = h_text
        for conv_layer in self.conv_stack:
            x = conv_layer(x, mask=mask)

        # Autoregressive Prediction
        p_0 = self.init_state(x)

        outputs = [p_0]
        for i in range(x.shape[1]):
            p_input = torch.cat((x[:, i], outputs[-1]), dim=-1) # [B, 3 * d_model]
            self.gru_hidden = self.gru_cell(p_input, self.gru_hidden) # [B, 2 * d_model]
            outputs.append(self.gru_hidden)
        outputs = torch.stack(outputs[1:], dim=1) # [B, src_len, 2 * d_model]

        # GMM-MDN
        w, sigma, mu = self.gmm_mdn(outputs)
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), 0 if self.training else 1e-9) # 1e-9 for categorical sampling
            sigma = sigma.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
            mu = mu.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return w, sigma, mu

    @staticmethod
    def sample(w, sigma, mu, mask=None):
        """ Draw samples from a GMM-MDN 
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        mask -- [B, src_len]
        output -- [B, src_len, out_features]
        """
        from torch.distributions import Categorical
        batch = []
        for i in range(w.shape[1]):
            w_i, sigma_i, mu_i = w[:, i], sigma[:, i], mu[:, i]
            ws = Categorical(w_i).sample().view(w_i.size(0), 1, 1)
            # Choose a random sample, one randn for batch X output dims
            # Do a (output dims)X(batch size) tensor here, so the broadcast works in
            # the next step, but we have to transpose back.
            gaussian_noise = torch.randn(
                (sigma_i.size(2), sigma_i.size(0)), requires_grad=False).to(w.device)
            variance_samples = sigma_i.gather(1, ws).detach().squeeze()
            mean_samples = mu_i.detach().gather(1, ws).squeeze()
            batch.append((gaussian_noise * variance_samples + mean_samples).transpose(0, 1))
        output = torch.stack(batch, dim=1)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output


class ReferenceEncoder(nn.Module):
    """ Reference Mel Encoder """

    def __init__(self,
                 n_mel_channels: int,
                 conv_filters: List[int],
                 conv_kernel_size: int,
                 conv_stride: int,
                 gru_size: int,
                 dropout_rate: float):
        super(ReferenceEncoder, self).__init__()

        self._dropout_rate = dropout_rate
        self.n_mel_channels = n_mel_channels
        filters = [1] + conv_filters
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(in_channels=filters[0],
                             out_channels=filters[0 + 1],
                             kernel_size=conv_kernel_size,
                             stride=conv_stride,
                             padding='same', with_r=True)]
        convs2 = [nn.Conv2d(in_channels=filters[i],
                            out_channels=filters[i + 1],
                            kernel_size=conv_kernel_size,
                            stride=conv_stride,
                            padding='same') for i in range(1, len(conv_filters))]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=n_filters) for n_filters in conv_filters])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, len(conv_filters))
        self.gru = nn.GRU(input_size=conv_filters[-1] * out_channels,
                          hidden_size=gru_size,
                          batch_first=True)

    def forward(self, inputs, mask=None):
        """
        inputs --- [N, Ty/r, n_mels*r]
        outputs --- [N, E//2]
        """
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
            out = F.dropout2d(out, p=self._dropout_rate)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0)

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # memory --- [N, Ty, E//2], out --- [1, N, E//2]

        return memory, out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class PhonemeLevelProsodyEncoder(nn.Module):
    """ Phoneme-level Prosody Encoder """

    def __init__(self, preprocess_config, model_config):
        super(PhonemeLevelProsodyEncoder, self).__init__()

        self.E = model_config["transformer"]["encoder_hidden"]
        self.d_q = self.d_k = model_config["transformer"]["encoder_hidden"]
        bottleneck_size = model_config["prosody_modeling"]["liu2021"]["bottleneck_size_p"]
        ref_enc_gru_size = model_config["prosody_modeling"]["liu2021"]["ref_enc_gru_size"]
        ref_attention_dropout = model_config["prosody_modeling"]["liu2021"]["ref_attention_dropout"]

        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.linears = nn.ModuleList([
            LinearNorm(in_dim, self.E, bias=False)
            for in_dim in (self.d_q, self.d_k)
        ])
        self.encoder_prj = nn.Linear(ref_enc_gru_size, self.E * 2)
        self.dropout = nn.Dropout(ref_attention_dropout)
        self.encoder_bottleneck = nn.Linear(self.E, bottleneck_size)

    def forward(self, x, text_lengths, src_mask, mels, mels_lengths, mel_mask):
        '''
        x --- [N, seq_len, encoder_embedding_dim]
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, bottleneck_size]
        attn --- [N, seq_len, ref_len], Ty/r = ref_len
        '''
        embedded_prosody, _ = self.encoder(mels, mel_mask)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Obtain k and v from prosody embedding
        k, v = torch.split(embedded_prosody, self.E, dim=-1) # [N, Ty, E] * 2

        # Get attention mask
        src_len, mel_len = x.shape[1], mels.shape[1]
        text_mask = src_mask.unsqueeze(-1).expand(-1, -1, mel_len) # [batch, seq_len, mel_len]
        mels_mask = mel_mask.unsqueeze(1).expand(-1, src_len, -1) # [batch, seq_len, mel_len]

        # Attention
        q, k = [linear(vector) for linear, vector in zip(self.linears, (x, k))]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # [N, seq_len, ref_len]
        attn = attn.masked_fill(mels_mask, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = attn.masked_fill(text_mask, 0.)
        out = self.encoder_bottleneck(torch.bmm(attn, v)) # [N, seq_len, bottleneck_size]
        out = out.masked_fill(src_mask.unsqueeze(-1), 0.)

        return out, attn


class STL(nn.Module):
    """ Style Token Layer """

    def __init__(self,
                 d_model: int,
                 n_tokens: int):
        super(STL, self).__init__()

        self.embed = nn.Parameter(torch.FloatTensor(n_tokens, d_model))
        
        self._input_ref_proj = nn.Linear(d_model, d_model // 2)

        self.attention = StyleEmbedAttention(
            query_dim=d_model // 2, key_dim=d_model, num_units=d_model, num_heads=1)

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, encoded_reference: torch.Tensor):
        batch_size = encoded_reference.size(0)
        query = self._input_ref_proj(encoded_reference).unsqueeze(1)  # [N, 1, E//2]

        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(
            batch_size, -1, -1)  # [N, token_num, E // num_heads]

        # Weighted sum
        emotion_embed_soft = self.attention(query, keys_soft)

        return emotion_embed_soft


class StyleEmbedAttention(nn.Module):
    """ StyleEmbedAttention """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(StyleEmbedAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim,
                               out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key_soft):
        """
        input:
            query --- [N, T_q, query_dim]
            key_soft --- [N, T_k, key_dim]
        output:
            out --- [N, T_q, num_units]
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        out_soft = scores_soft = None
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key_soft)  # [N, T_k, num_units]

        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores_soft = torch.matmul(
            querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores_soft = scores_soft / (self.key_dim ** 0.5)
        scores_soft = F.softmax(scores_soft, dim=3)

        # out = score * V
        # [h, N, T_q, num_units/h]
        out_soft = torch.matmul(scores_soft, values)
        out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(
            0)  # [N, T_q, num_units]

        return out_soft #, scores_soft


class UtteranceLevelProsodyEncoder(nn.Module):
    """ Utterance-level Prosody Encoder """

    def __init__(self,
                 encoder_hidden_size: int,
                 input_mel_channels: int,
                 ref_enc_filters: List[int],
                 ref_enc_gru_size: int,
                 ref_enc_kernel_size: int,
                 ref_enc_stride: int,
                 n_gst_tokens: int,
                 dropout_rate: float):
        super(UtteranceLevelProsodyEncoder, self).__init__()

        self.encoder = ReferenceEncoder(input_mel_channels,
                                        ref_enc_filters,
                                        ref_enc_kernel_size,
                                        ref_enc_stride,
                                        ref_enc_gru_size)
        self.stl = STL(encoder_hidden_size, n_gst_tokens)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, mels, mel_mask):
        '''
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, E]
        '''
        _, embedded_prosody = self.encoder(mels, mel_mask)

        # Style Token
        out = self.stl(embedded_prosody)
        out = self.dropout(out)

        return out


class ParallelProsodyPredictor(nn.Module):
    """ Parallel Prosody Predictor """

    def __init__(self, model_config, phoneme_level=True):
        super(ParallelProsodyPredictor, self).__init__()

        self.phoneme_level = phoneme_level
        self.E = model_config["transformer"]["encoder_hidden"]
        self.input_size = self.E
        self.filter_size = self.E
        self.conv_output_size = self.E
        self.kernel = model_config["prosody_modeling"]["liu2021"]["predictor_kernel_size"]
        self.dropout = model_config["prosody_modeling"]["liu2021"]["predictor_dropout"]
        bottleneck_size = model_config["prosody_modeling"]["liu2021"]["bottleneck_size_p"] if phoneme_level else\
                          model_config["prosody_modeling"]["liu2021"]["bottleneck_size_u"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=1,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )
        self.gru = nn.GRU(input_size=self.E,
                          hidden_size=self.E//2,
                          batch_first=True,
                          bidirectional=True,)
        self.predictor_bottleneck = nn.Linear(self.E, bottleneck_size)

    def forward(self, x):
        """
        x --- [N, src_len, hidden]
        """
        x = self.conv_layer(x)

        self.gru.flatten_parameters()
        memory, out = self.gru(x)

        if self.phoneme_level:
            pv_forward = memory[:, :, :self.E//2]
            pv_backward = memory[:, :, self.E//2:]
            prosody_vector = torch.cat((pv_forward, pv_backward), dim=-1)
        else:
            out = out.transpose(0, 1)
            prosody_vector = torch.cat((out[:, 0], out[:, 1]), dim=-1).unsqueeze(1)
        prosody_vector = self.predictor_bottleneck(prosody_vector)

        return prosody_vector


class NonParallelProsodyPredictor(nn.Module):
    """ Non-parallel Prosody Predictor inspired by Du et al., 2021 """

    def __init__(self, model_config, phoneme_level=True):
        super(NonParallelProsodyPredictor, self).__init__()

        self.phoneme_level = phoneme_level
        # self.E = model_config["transformer"]["encoder_hidden"]
        self.d_model = model_config["transformer"]["encoder_hidden"]
        kernel_size = model_config["prosody_modeling"]["liu2021"]["predictor_kernel_size"]
        dropout = model_config["prosody_modeling"]["liu2021"]["predictor_dropout"]
        bottleneck_size = model_config["prosody_modeling"]["liu2021"]["bottleneck_size_p"] if phoneme_level else\
                          model_config["prosody_modeling"]["liu2021"]["bottleneck_size_u"]
        self.conv_stack = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=kernel_size[i],
                    dropout=dropout,
                    normalization=nn.LayerNorm,
                    transpose=True,
                )
                for i in range(2)
            ]
        )
        self.gru_cell = nn.GRUCell(
            self.d_model + 2 * self.d_model,
            2 * self.d_model,
        )
        self.predictor_bottleneck = nn.Linear(2 * self.d_model, bottleneck_size)

    def init_state(self, x):
        """
        x -- [B, src_len, d_model]
        p_0 -- [B, 2 * d_model]
        self.gru_hidden -- [B, 2 * d_model]
        """
        B, _, d_model = x.shape
        p_0 = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        self.gru_hidden = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        return p_0

    def forward(self, h_text, mask=None):
        """
        h_text -- [B, src_len, d_model]
        mask -- [B, src_len]
        outputs -- [B, src_len, 2 * d_model]
        """
        x = h_text
        for conv_layer in self.conv_stack:
            x = conv_layer(x, mask=mask)

        # Autoregressive Prediction
        p_0 = self.init_state(x)

        outputs = [p_0]
        for i in range(x.shape[1]):
            p_input = torch.cat((x[:, i], outputs[-1]), dim=-1) # [B, 3 * d_model]
            self.gru_hidden = self.gru_cell(p_input, self.gru_hidden) # [B, 2 * d_model]
            outputs.append(self.gru_hidden)
        outputs = torch.stack(outputs[1:], dim=1) # [B, src_len, 2 * d_model]

        if mask is not None:
            outputs = outputs.masked_fill(mask, 0.0)

        if self.phoneme_level:
            prosody_vector = outputs # [B, src_len, 2 * d_model]
        else:
            prosody_vector = torch.mean(outputs, dim=1, keepdim=True) # [B, 1, 2 * d_model]
        prosody_vector = self.predictor_bottleneck(prosody_vector)

        return prosody_vector
    

@torch.no_grad()
def mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:

    batch_size = lengths.size(0)

    max_len = torch.max(lengths).item()
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=lengths.device)

    for i in range(batch_size):
        mask[i, :lengths[i]] = 1

    return mask


class VarianceAdaptor(nn.Module):
    """Variance Adaptor.
    
    The implementation has been reworked and supports the following setup:
        - Unsupervised alignment learning.
        - Explicit utterance-level prosody modelling (pitch and energy quantized into bins).
    """

    def __init__(self,
                 d_model: int,
                 multi_speaker: bool,
                 use_utterance_level_prosody: bool,
                 predictor_n_layers: int,
                 predictor_n_int_channels: int,
                 predictor_kernel_size: int,
                 dropout_rate: float,
                 spk_emb_dim: int):
        
        super().__init__()

        self._multi_speaker = multi_speaker
        self._use_utterance_level_prosody = use_utterance_level_prosody

        self.length_regulator = LengthRegulator()

        self.duration_predictor = ProsodicFeaturesPredictor(
                d_model,
                n_chans=predictor_n_int_channels,
                n_layers=predictor_n_layers,
                dropout_rate=dropout_rate,
                kernel_size=predictor_kernel_size)
        
        self._spk_emb_enc = None

        if self._multi_speaker:
            self._spk_emb_enc = nn.Linear(spk_emb_dim, d_model)

        self._pitch_transform = None
        self._pitch_predictor = None
        self._energy_transform = None
        self.energy_predictor = None
        
        if self._use_utterance_level_prosody:

            self._pitch_transform = nn.Sequential(
                torch.nn.Conv1d(in_channels=1,
                                out_channels=d_model,
                                kernel_size=3,
                                padding='same'),
                nn.ReLU(),
                torch.nn.Conv1d(in_channels=d_model,
                                out_channels=d_model,
                                kernel_size=3,
                                padding='same'),
                )

            self._pitch_predictor = ProsodicFeaturesPredictor(
                idim=d_model,
                n_layers=predictor_n_layers,
                n_chans=predictor_n_int_channels,
                kernel_size=predictor_kernel_size,
                dropout_rate=dropout_rate)
            
            self._energy_transform = nn.Sequential(
                torch.nn.Conv1d(in_channels=1,
                                out_channels=d_model,
                                kernel_size=3,
                                padding='same'),
                nn.ReLU(),
                torch.nn.Conv1d(in_channels=d_model,
                                out_channels=d_model,
                                kernel_size=3,
                                padding='same'),
                )

            self.energy_predictor = ProsodicFeaturesPredictor(
                idim=d_model,
                n_layers=predictor_n_layers,
                n_chans=predictor_n_int_channels,
                kernel_size=predictor_kernel_size,
                dropout_rate=dropout_rate)

    def forward(
        self,
        phoneme_repr: torch.Tensor,
        phonemes_length: torch.Tensor,

        # Only if using utterance-level prosody
        pitch_possible_values: Optional[torch.Tensor],
        energy_possible_values: Optional[torch.Tensor],
        # (training)
        pitch_target: Optional[torch.Tensor],
        energy_target: Optional[torch.Tensor],

        # Only if multi-speaker
        speaker_embedding: Optional[torch.Tensor],

        # (training)
        explicit_durations: Optional[torch.Tensor]):
        
        model_output = {}

        if not self._multi_speaker:
            assert speaker_embedding is None

        else:
            assert speaker_embedding is not None

        train_only_inputs = [explicit_durations]

        if not self._use_utterance_level_prosody:
            assert all(el is None for el in (
                pitch_possible_values,
                energy_possible_values,
                pitch_target,
                energy_target,
            ))

        else:
            assert all(el is not None for el in (
                pitch_possible_values,
                energy_possible_values,
            ))

            train_only_inputs.extend([pitch_target, energy_target])
        
        if any(el is None for el in train_only_inputs):
            inference_mode = True
            assert all(el is None for el in train_only_inputs)    
        
        else:
            inference_mode = False
            assert all(el is not None for el in train_only_inputs)

        outputs = phoneme_repr

        if self._multi_speaker:
            assert speaker_embedding is not None

            spk_emb_encoded = self._spk_emb_enc(speaker_embedding)

            outputs = outputs + spk_emb_encoded.unsqueeze(1).expand(
                -1, phoneme_repr.shape[1], -1
            )

        predicted_durations = self.duration_predictor(outputs.detach())
        model_output['predicted_durations'] = predicted_durations

        if not inference_mode:
            mel_lengths = explicit_durations.sum(dim=1).long()
            outputs, _ = self.length_regulator(outputs,
                                               explicit_durations,
                                               mel_lengths.max())

        else:
            duration_rounded = torch.clamp(torch.round(predicted_durations), min=0)
            duration_rounded *= mask_from_lengths(phonemes_length)

            mel_lengths = duration_rounded.sum(dim=1).long()
            outputs, _ = self.length_regulator(outputs,
                                               duration_rounded.long(),
                                               max_len=mel_lengths.max())

        if self._use_utterance_level_prosody:
            predicted_pitch = self._pitch_predictor(outputs.detach())
            predicted_energy = self.energy_predictor(outputs.detach())

            model_output['predicted_pitch'] = predicted_pitch
            model_output['predicted_energy'] = predicted_energy

            if not inference_mode:
                chosen_pitch = pitch_target
                chosen_energy = energy_target

            else:
                chosen_pitch = predicted_pitch
                chosen_energy = predicted_energy

            pitch_indices = torch.bucketize(chosen_pitch, pitch_possible_values)
            energy_indices = torch.bucketize(chosen_energy, energy_possible_values)

            pitch_quant = pitch_possible_values[pitch_indices - 1]
            energy_quant = energy_possible_values[energy_indices - 1]

            if not inference_mode:
                model_output['target_pitch_quant'] = pitch_quant
                model_output['target_energy_quant'] = energy_quant

            pitch_enc = self._pitch_transform(pitch_quant.unsqueeze(-1).transpose(1, 2))
            energy_enc = self._energy_transform(energy_quant.unsqueeze(-1).transpose(1, 2))

            outputs = outputs + pitch_enc.transpose(1, 2) + energy_enc.transpose(1, 2)

        model_output['output'] = outputs

        return model_output

class MaskedSoftmax(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()

        self._softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x, mask):
		
        if mask is not None:
            x.data.masked_fill_(~mask, -float("inf"))

        smax = self._softmax(x)

        if mask is not None:
            smax.data.masked_fill_(~mask, 0.0)

        return smax
    
class MaskedLogSoftmax(torch.nn.Module):
    
    def __init__(self, dim=-1):
        super().__init__()

        self._softmax = torch.nn.LogSoftmax(dim=dim)

    def forward(self, x, mask):
		
        if mask is not None:
            x.data.masked_fill_(~mask, -float("inf"))

        smax = self._softmax(x)

        if mask is not None:
            smax.data.masked_fill_(~mask, -float("inf"))

        return smax

class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, 
                n_mel_channels,
                n_att_channels,
                n_text_channels,
                temperature,
                multi_speaker):
        super().__init__()
        self.temperature = temperature
        self.softmax = MaskedSoftmax(dim=2)
        self.log_softmax = MaskedLogSoftmax(dim=2)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=False)

        if attn_prior is not None:
            #print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn, mask) + (torch.log(attn_prior + 1e-8) * mask)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        attn = self.softmax(attn, mask)  # softmax along T2
        return attn, attn_logprob


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class ProsodicFeaturesPredictor(torch.nn.Module):
    """Predicts prosody features from stratched textual features."""

    def __init__(self,
                 idim: int,
                 n_layers: int,
                 n_chans: int,
                 kernel_size: int,
                 dropout_rate: float):
        """Initilize prosody predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        
        super().__init__()
        
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(idim, n_chans),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )

        for _ in range(n_layers):
            self.conv += [torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_chans),
                torch.nn.Conv1d(n_chans, n_chans, kernel_size, stride=1, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Dropout1d(dropout_rate)
            )]

        self._postnet = torch.nn.Linear(n_chans, 1)

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        xs = self._prenet(xs)

        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        
        return self._postnet(xs.transpose(1, -1)).squeeze(-1) # [B, T]

