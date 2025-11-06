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

class _LinguisticAwareEncoder(nn.Module):
    """Encodes phonemes and linguistic information.
    
    The encoded linguistic information is used to enhance the reference encoder.
    """

    def __init__(self,
                 phonemes_vocab_size: int,
                 hidden_size: int,
                 linguistic_features_dim: int,
                 dropout_rate: float):
        
        super().__init__()

        self._phoneme_encoder = torch.nn.Sequential(
            torch.nn.Embedding(phonemes_vocab_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout_rate)
        )

        self._ling_info_encoder = torch.nn.Sequential(
            torch.nn.Linear(linguistic_features_dim, hidden_size),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self,
                phoneme_ids: torch.Tensor,
                linguistic_features: torch.Tensor,
                phoneme_to_spec_indices: torch.Tensor):

        bsize = phoneme_ids.size(0)

        encoded_phonemes = self._phoneme_encoder(phoneme_ids)
        encoded_ling_info = self._ling_info_encoder(linguistic_features)

        output = encoded_phonemes + encoded_ling_info

        return output[torch.arange(bsize).unsqueeze(1), phoneme_to_spec_indices]


class _ReferenceEncoder(nn.Module):
    """ Reference Mel Encoder """

    def __init__(self,
                 d_model: int,
                 n_mel_channels: int,
                 conv_filters: List[int],
                 conv_kernel_size: int,
                 conv_stride: List[int],
                 conv_padding: List[int],
                 gru_size: int,
                 dropout_rate: float,
                 linguistic_aware_encoder: Optional[_LinguisticAwareEncoder]):
        """
        Args:
            d_model: Hidden size of the model.
            n_mel_channels: Number of mel channels.
            conv_filters: Number of filters for each conv layer.
            conv_kernel_size: Kernel size for conv layers.
            conv_stride: Stride size for conv layers. E.g. [1, 2] to preserve time resolution.
            conv_padding: Padding size for conv layers.
            gru_size: Hidden size of the GRU layer. Determines the size of the bottleneck.
            dropout_rate: Dropout rate for conv layers.
            linguistic_aware_encoder: Encoder for linguistic information to enhance the reference encoder.
        """
        
        super().__init__()

        self._dropout_rate = dropout_rate
        self.n_mel_channels = n_mel_channels
        filters = [1] + conv_filters
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(in_channels=filters[0],
                             out_channels=filters[0 + 1],
                             kernel_size=conv_kernel_size,
                             stride=conv_stride,
                             padding=conv_padding, with_r=True)]
        convs2 = [nn.Conv2d(in_channels=filters[i],
                            out_channels=filters[i + 1],
                            kernel_size=conv_kernel_size,
                            stride=conv_stride,
                            padding=conv_padding) for i in range(1, len(conv_filters))]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=n_filters) for n_filters in conv_filters])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, len(conv_filters))
        
        self._conv_postnet = torch.nn.Linear(conv_filters[-1] * out_channels, d_model)
        
        self.gru = nn.GRU(input_size=d_model,
                          hidden_size=gru_size,
                          batch_first=True)

        self._post_net = nn.Linear(gru_size, d_model)

        self._ling_aware_encoder = linguistic_aware_encoder

    def forward(self,
                spectrogram,
                spectrogram_length,
                return_global: bool,
                return_local: bool,
                phoneme_ids: Optional[torch.Tensor],
                linguistic_features: Optional[torch.Tensor],
                phoneme_to_spec_indices: Optional[torch.Tensor]):
        """
        Args:
            spectrogram: Input spectrogram (transposed). (B, T, n_mel_channels)
            spectrogram_length: Length of the input spectrogram. (B,)
        """

        N = spectrogram.size(0)
        out = spectrogram.view(N, 1, -1, self.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
            out = F.dropout2d(out, p=self._dropout_rate)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        out = self._conv_postnet(out)

        if self._ling_aware_encoder is not None:
            assert phoneme_ids is not None
            assert linguistic_features is not None
            assert phoneme_to_spec_indices is not None

            encoded_ling_info = self._ling_aware_encoder(
                phoneme_ids,
                linguistic_features,
                phoneme_to_spec_indices
            )

            out = out + encoded_ling_info

        packed_sequence = nn.utils.rnn.pack_padded_sequence(out,
                                                            spectrogram_length.cpu(),
                                                            batch_first=True,
                                                            enforce_sorted=False)

        self.gru.flatten_parameters()
        packed_output, global_ref_embedding = self.gru(packed_sequence)  # memory --- [N, Ty, E//2], out --- [1, N, E//2]

        local_ref_embeddings, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if return_global and return_local:
            return (self._post_net(local_ref_embeddings),
                    self._post_net(global_ref_embedding.squeeze(0)))

        elif return_global:
            return self._post_net(global_ref_embedding.squeeze(0))
        
        elif return_local:
            return self._post_net(local_ref_embeddings)
        
        raise ValueError("At least one of return_global or return_local must be True.")

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class _STL(nn.Module):
    """ Style Token Layer """

    def __init__(self,
                 d_model: int,
                 n_tokens: int):
        super().__init__()

        self.embed = nn.Parameter(torch.FloatTensor(n_tokens, d_model))
        
        # self._input_ref_proj = nn.Linear(d_model, d_model // 2)

        self.attention = _StyleEmbedAttention(
            query_dim=d_model, key_dim=d_model, num_units=d_model)

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, encoded_reference: torch.Tensor):
        batch_size = encoded_reference.size(0)
        query = encoded_reference.unsqueeze(1)

        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(
            batch_size, -1, -1)  # [N, token_num, E // num_heads]

        # Weighted sum
        embedding, embedding_weights = self.attention(query, keys_soft)

        return embedding, embedding_weights


class _StyleEmbedAttention(nn.Module):
    """ StyleEmbedAttention """

    def __init__(self, query_dim, key_dim, num_units):
        super().__init__()
        self.num_units = num_units
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

        out_soft = scores_soft = None
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key_soft)  # [N, T_k, num_units]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores_soft = torch.matmul(
            querys, keys.transpose(-2, -1))  # [h, N, T_q, T_k]
        scores_soft = scores_soft / (self.key_dim ** 0.5)
        scores_soft = F.softmax(scores_soft, dim=-1)

        # out = score * V
        # [h, N, T_q, num_units/h]
        out_soft = torch.matmul(scores_soft, values)

        return out_soft.squeeze(-2), scores_soft.squeeze(-2)


class UtteranceLevelProsodyEncoder(nn.Module):
    """ Utterance-level Prosody Encoder """

    def __init__(self,
                 encoder_hidden_size: int,
                 input_mel_channels: int,
                 ref_enc_filters: List[int],
                 ref_enc_gru_size: int,
                 ref_enc_kernel_size: int,
                 ref_enc_stride: List[int],
                 ref_enc_conv_padding: List[int],
                 n_gst_tokens: int,
                 dropout_rate: float,
                 linguistic_aware: bool,
                 phonemes_vocab_size: int,
                 linguistic_features_dim: int):
        super(UtteranceLevelProsodyEncoder, self).__init__()

        linguistic_aware_encoder = None

        if linguistic_aware:
            linguistic_aware_encoder = _LinguisticAwareEncoder(
                phonemes_vocab_size=phonemes_vocab_size,
                hidden_size=encoder_hidden_size,
                linguistic_features_dim=linguistic_features_dim,
                dropout_rate=dropout_rate
            )

        self.encoder = _ReferenceEncoder(encoder_hidden_size,
                                        input_mel_channels,
                                        ref_enc_filters,
                                        ref_enc_kernel_size,
                                        ref_enc_stride,
                                        ref_enc_conv_padding,
                                        ref_enc_gru_size,
                                        dropout_rate,
                                        linguistic_aware_encoder)
        self.stl = _STL(encoder_hidden_size, n_gst_tokens)
        self.dropout = nn.Dropout(dropout_rate)

    @property
    def is_linguistic_aware(self) -> bool:
        return self.encoder._ling_aware_encoder is not None

    def forward(self, 
                spectrogram: torch.Tensor,
                spectrogram_length: torch.Tensor,

                phoneme_ids: Optional[torch.Tensor],
                linguistic_features: Optional[torch.Tensor],
                phoneme_spec_indices: Optional[torch.Tensor]):
        """Encodes input spectrogram into GST-based style embedding.
        
        Args:
            spectrogram: Input spectrogram. (B, n_mel_channels, T)
            spectrogram_length: Lengths of the input spectrogram. (B,)
        """

        ref_emb_global = self.encoder(spectrogram.transpose(1, 2),
                                      spectrogram_length,
                                      return_global=True,
                                      return_local=False,
                                      phoneme_ids=phoneme_ids,
                                      linguistic_features=linguistic_features,
                                      phoneme_to_spec_indices=phoneme_spec_indices)

        gst_embedding, gst_weights = self.stl(ref_emb_global)

        return self.dropout(gst_embedding), gst_weights
    

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
            expanded = torch.repeat_interleave(batch, expand_target, dim=0)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

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

