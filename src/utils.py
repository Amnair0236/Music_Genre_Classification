import librosa
import numpy as np
import cv2

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, duration=30)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize EXACTLY like images (0–1)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    mel_db = cv2.resize(mel_db, (128, 128))
    mel_db = np.stack([mel_db] * 3, axis=-1)

    return np.expand_dims(mel_db, axis=0)
