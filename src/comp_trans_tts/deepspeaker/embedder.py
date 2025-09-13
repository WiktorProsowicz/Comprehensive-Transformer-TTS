"""Contains definition of a model producing speaker embedding using pre-trained DeepSpeaker."""

import os
import subprocess

import gdown
import onnxruntime
import logging
import numpy as np


PRETRAINED_MODEL_URL = 'https://drive.google.com/uc?id=10R9dhgX6IJQ3Nf7D0UdlZpwqL1AJn3Zi'


class DeepSpeakerEmbedder:
    """Produces embeddings using pre-trained DeepSpeaker model."""

    def __init__(self, model_persist_dir='~/.cache/comp_trans_tts'):
        """Inits the speaker embedder.

        Args:
            model_persist_dir: Directory the model will be cached in.
        """

        model_persist_dir = os.path.expanduser(model_persist_dir)
        os.makedirs(model_persist_dir, exist_ok=True)
        archive_path = os.path.join(model_persist_dir, 'deepspeaker_pretrained.tar.bz2')

        if not os.path.exists(os.path.join(model_persist_dir, 'deepspeaker_pretrained_onnx')):

            gdown.download(PRETRAINED_MODEL_URL, archive_path, quiet=False)

            try:
                subprocess.run(['bzip2', '-d', archive_path], check=True)

                tar_path = archive_path[:-4]

                subprocess.run(['tar', '-xf', tar_path, '-C', model_persist_dir], check=True)

                subprocess.run(['rm', tar_path], check=True)

            except subprocess.CalledProcessError as proc_err:
                logging.critical('Failed to download the phoneme alignments: %s', proc_err)

        self._onnx_session = onnxruntime.InferenceSession(
            os.path.join(model_persist_dir, 'deepspeaker_pretrained_onnx', 'deepspeaker.onnx')
        )

    def __call__(self, model_input: np.ndarray):
        """Calls the DeepSpeaker model and returns its output.

        Args:
            model_input: Input to the model without batch dimension.
        """

        return self._onnx_session.run(None, {'x': model_input[np.newaxis, :, :, :]})[0]
