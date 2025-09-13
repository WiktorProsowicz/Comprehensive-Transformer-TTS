"""Contains preprocessing function for DeepSpeaker model input.

Most functions and classes in this module are adapted from:
https://github.com/philipperemy/deep-speaker

If not stated otherwise, the code was copied in its original form.

In particular, the Audio class has been minimized.
"""

import logging
import os
from pathlib import Path

import librosa
import numpy as np
import numpy.random as npr
from python_speech_features import fbank


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NUM_FRAMES = 160
NUM_FBANKS = 64

def load_wav_for_deepseaker(filename):
    return sample_from_mfcc(read_mfcc(filename, SAMPLE_RATE), NUM_FRAMES).astype(np.float32)


def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = npr.choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)

def read_mfcc(input_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
    # TODO: could use trim_silence() here or a better VAD.
    audio_voice_only = audio[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc


def extract_speaker_and_utterance_ids(filename: str):  # LIBRI.
    # 'audio/dev-other/116/288045/116-288045-0000.flac'
    speaker, _, basename = Path(filename).parts[-3:]
    filename.split('-')
    utterance = os.path.splitext(basename.split('-', 1)[-1])[0]
    assert basename.split('-')[0] == speaker
    return speaker, utterance


# Note: This class has been stripped from most of its original functionality.
class Audio:

    @staticmethod
    def read(filename, sample_rate):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio


def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
