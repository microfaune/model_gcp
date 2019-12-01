import base64

import googleapiclient.discovery
import numpy as np

from scipy.io import wavfile
import librosa

# fs, data = wavfile.read('test.wav')
# print(data)
# print(data.min(), data.max())

enc = base64.b64encode(open("test1.wav", "rb").read()).decode()

PROJECT_ID = "gifted-honor-259919"
MODEL_NAME = 'birdDetectionModel'
VERSION_NAME = 'v1'

# new_data = np.frombuffer(base64.b64decode(enc), dtype=np.int16)
# print(new_data)
# print(new_data.min(), new_data.max())
# wavfile.write('test1.wav', fs, new_data)

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': enc}
).execute()

if 'error' in response:
    print(response)
    raise RuntimeError(response['error'])
else:
    print(response['predictions'])
    # new_data = np.frombuffer(base64.b64decode(response['predictions']), dtype=np.int16)
    # print(new_data)


def compute_features(audio_signals):
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

        x = create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X


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

    return S


# data = np.frombuffer(base64.b64decode(enc), dtype=np.int16)
# data = data.astype(np.float32)
# X = compute_features([data])
# print(X)
