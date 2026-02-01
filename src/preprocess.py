import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "data/genres"

# Output folder for spectrograms
OUTPUT_PATH = "data/spectrograms"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def create_mel_spectrogram(audio_path, save_path):
    try:
        # Load .au or .wav files
        y, sr = librosa.load(audio_path, duration=30)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Save spectrogram image
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, cmap='magma')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Loop through all genres and .au files
for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    # Skip if not a folder
    if not os.path.isdir(genre_path):
        continue

    save_dir = os.path.join(OUTPUT_PATH, genre)
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith(".au"):   # UPDATED from .wav to .au
            audio_file = os.path.join(genre_path, file)
            save_file = os.path.join(save_dir, file.replace(".au", ".png"))

            print(f"Processing: {audio_file}")
            create_mel_spectrogram(audio_file, save_file)

print("\nDone! Check the 'data/spectrograms' folder.")
