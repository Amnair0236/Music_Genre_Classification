import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/cnn_genre_model.h5")

# Path to spectrogram image
IMAGE_PATH = "data/spectrograms/jazz/jazz.00000.png"

# Read and preprocess image 
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
input_img = np.expand_dims(img / 255.0, axis=0)

# To find the last Conv2D layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# Create Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_img)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)

# Global average pooling on gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Weight the convolution outputs
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (heatmap.max() + 1e-8)
heatmap = np.uint8(255 * heatmap)

# Resize heatmap to image size
heatmap = cv2.resize(heatmap, (128, 128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

# IMPROVED OVERLAY CLARITY
overlay = cv2.addWeighted(
    img,       # original spectrogram
    0.65,       # lower image opacity
    heatmap,   # Grad-CAM heatmap
    0.5,       # stronger heatmap visibility
    0
)

# Display results
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM Visualization")
plt.show()
