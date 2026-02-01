import os
import sys
import cv2
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Music Genre Classification with Grad-CAM")

MODEL_PATH = "models/cnn_genre_model.h5"
TEMP_DIR = "data/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- FUNCTIONS ----------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, duration=30)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = cv2.resize(mel_db, (128, 128))
    mel_db = np.stack([mel_db]*3, axis=-1)

    return np.expand_dims(mel_db, axis=0)

def generate_gradcam(model, img_array):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    original = img_array[0] * 255
    original = original.astype(np.uint8)

    overlay = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)
    return overlay

# ---------------- UI ----------------
st.title("Music Genre Classification with Grad-CAM")
st.write("Upload an audio file to hear it, predict its genre, and visualize Grad-CAM")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "au"]
)

if uploaded_file is not None:
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Audio Player
    st.subheader("Audio Preview")
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Genre & Show Grad-CAM"):
        with st.spinner("Analyzing audio..."):
            spec = audio_to_spectrogram(file_path)
            prediction = model.predict(spec)[0]

            idx = int(np.argmax(prediction))
            genre = GENRES[idx]
            confidence = prediction[idx] * 100

            gradcam_img = generate_gradcam(model, spec)

        st.success(f"🎶 Predicted Genre: **{genre}** ({confidence:.2f}%)")

        st.subheader("🔍 Grad-CAM Explanation")
        st.image(
            gradcam_img,
            caption="Grad-CAM highlights important spectrogram regions",
            width=700
        )
