import os
import json
import base64

import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow import math
from scipy.io import wavfile
from scipy import signal
import librosa


class MyPredictor(object):
    def __init__(self, model):
        self._model = model

    def predict(self, instances, **kwargs):
        preds_global, preds_local = self.predict_on_bytes(instances)
        return json.dumps(preds_local[1].tolist())

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.h5')
        # model = tf.keras.models.load_model(model_path)
        model = create_model()
        print(model_path)
        model.load_weights(model_path)
        return cls(model)

    def compute_features(self, audio_signals):
        """Compute features on audio signals.

        Parameters
        ----------
        audio_signals: list
            Audio signals of possibly various lengths.

        Returns
        -------
        X: list
            Features for each audio signal
        """
        X = []
        for data in audio_signals:

            x = self.create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                                 hop_len=1024).transpose()
            X.append(x[..., np.newaxis].astype(np.float32)/255)
        return X

    def bytes_to_numpy(bytes_data):
        data = np.frombuffer(base64.b64decode(bytes_data), dtype=np.int16)
        return data

    def predict_on_bytes(self, bytes_data):
        """Detect bird presence in wav file.

        Parameters
        ----------
        bytes: str
            Audio bytes.

        Returns
        -------
        score: float
            Prediction score of the classifier on the whole sequence
        local_score: array-like
            Time step prediction score
        """
        data = self.bytes_to_numpy(bytes_data)
        X = self.compute_features([data])
        scores, local_scores = self.predict_model(np.array(X))
        return scores[0], local_scores[0]

    def predict_model(self, X):
        """Predict bird presence on spectograms.

        Parameters
        ----------
        X: array-like
            List of features on which to run the model.

        Returns
        -------
        scores: array-like
            Prediction scores of the classifier on each audio signal
        local_scores: array-like
            Step-by-step  prediction scores for each audio signal
        """
        scores = []
        local_scores = []
        for x in X:
            s, local_s = self._model.predict(x[np.newaxis, ...])
            scores.append(s[0])
            local_scores.append(local_s.flatten())
        scores = np.array(s)
        return scores, local_scores

    def load_wav(path, decimate=None):
        """Load audio data.

            Parameters
            ----------
            path: str
                Wav file path.
            decimate: int
                If not None, downsampling by a factor of `decimate` value.

            Returns
            -------
            S: array-like
                Array of shape (Mel bands, time) containing the spectrogram.
        """
        fs, data = wavfile.read(path)

        data = data.astype(np.float32)

        if decimate is not None:
            data = signal.decimate(data, decimate)
            fs /= decimate

        return fs, data

    def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
        """Compute the Mel spectrogram from audio data.

            Parameters
            ----------
            data: array-like
                Audio data.
            fs: int
                Sampling frequency in Hz.
            n_mels: int
                Number of Mel bands to generate.
            n_fft: int
                Length of the FFT window.
            hop_len: int
                Number of samples between successive frames.

            Returns
            -------
            S: array-like
                Array of shape (Mel bands, time) containing the spectrogram.
        """
        # Calculate spectrogram
        S = librosa.feature.melspectrogram(
          data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        S = S.astype(np.float32)

        # Convert power to dB
        S = librosa.power_to_db(S)


def create_model():
    """Create RNN model."""
    n_filter = 64

    spec = layers.Input(shape=[None, 40, 1], dtype=np.float32)
    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(spec)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((1, 2))(x)

    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((1, 2))(x)

    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filter, (3, 3), padding="same",
                      activation=None)(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((1, 2))(x)

    x = math.reduce_max(x, axis=-2)

    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, recurrent_activation='sigmoid'))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, recurrent_activation='sigmoid'))(x)

    x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
    local_pred = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"))(x)
    pred = math.reduce_max(local_pred, axis=-2)
    return keras.Model(inputs=spec, outputs=[pred, local_pred])
