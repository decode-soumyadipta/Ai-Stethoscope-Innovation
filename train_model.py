import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import difflib
import math
from collections import Counter

# Configuration
EXCEL_FILE = 'datasets/Data annotation.xlsx'
AUDIO_DIR = 'datasets/Audio Files'
MODEL_SAVE_PATH = 'models/model.h5'
SAMPLE_RATE = 16000
TARGET_AUDIO_DURATION = 2.0  # seconds
TARGET_LENGTH = 350
N_MELS = 128
N_MFCC = 20
LABELS_TO_DISEASES = {
    0: "Healthy",
    1: "Asthma",
    2: "COPD",
    3: "Heart Failure",
    4: "Lung Fibrosis"
}

# 1. Load metadata
df = pd.read_excel(EXCEL_FILE)
print("Columns in Excel:", df.columns.tolist())

# 2. List all audio files
wav_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
print(f"Found {len(wav_files)} wav files.")

# 3. Prepare data
X = []
y = []
used_files = set()
def norm(s):
    if pd.isna(s):
        return ''
    return str(s).replace(' ', '').lower().strip()

for idx, row in df.iterrows():
    # Normalize and check for missing values
    diagnosis = norm(row['Diagnosis'])
    sound_type = norm(row['Sound type'])
    location = norm(row['Location'])
    age = norm(row['Age'])
    gender = norm(row['Gender'])
    if not all([diagnosis, sound_type, location, age, gender]):
        print(f"Skipping row {idx} due to missing value(s).")
        continue

    # Try to find a matching file
    best_match = None
    best_score = 0
    for fname in wav_files:
        fname_norm = fname.replace(' ', '').lower()
        score = 0
        if diagnosis and diagnosis in fname_norm:
            score += 1
        if sound_type and sound_type in fname_norm:
            score += 1
        if location and location in fname_norm:
            score += 1
        if age and age in fname_norm:
            score += 1
        if gender and gender in fname_norm:
            score += 1
        if score > best_score:
            best_score = score
            best_match = fname
    print(f"Row {idx}: Best candidate: {best_match} (score {best_score})")
    if best_match and best_score >= 3:  # Lowered threshold
        audio_path = os.path.join(AUDIO_DIR, best_match)
        used_files.add(best_match)
        label = diagnosis
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            # Pad or trim audio
            target_len = int(SAMPLE_RATE * TARGET_AUDIO_DURATION)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            # Normalize
            audio = librosa.util.normalize(audio)
            # Mel-spectrogram
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS)
            log_S = librosa.power_to_db(S, ref=np.max)
            spec = librosa.util.fix_length(log_S, size=TARGET_LENGTH, axis=1)
            spec = spec[:N_MELS, :]
            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            mfcc = librosa.util.fix_length(mfcc, size=TARGET_LENGTH, axis=1)
            mfcc = mfcc[:N_MFCC, :]
            # Concatenate along feature axis (stack vertically)
            combined = np.concatenate([spec, mfcc], axis=0)  # shape: (N_MELS+N_MFCC, TARGET_LENGTH)
            X.append(combined)
            y.append(label)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    else:
        print(f"No good match for row {idx}: Diagnosis={diagnosis}, Sound type={sound_type}, Location={location}, Age={age}, Gender={gender}")

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} samples for training.")
print(f"Feature shape: {X.shape}")

# Print class counts before filtering
label_counts = Counter(y)
print("Class counts before filtering:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# Filter out classes with fewer than 2 samples
valid_labels = {label for label, count in label_counts.items() if count >= 2}
if not valid_labels:
    raise ValueError("No classes have at least 2 samples. Cannot proceed with training.")
mask = np.array([label in valid_labels for label in y])
X = X[mask]
y = y[mask]

# Print class counts after filtering
label_counts_after = Counter(y)
print("Class counts after filtering:")
for label, count in label_counts_after.items():
    print(f"  {label}: {count}")
    if count < 5:
        print(f"  [WARNING] Class '{label}' has only {count} samples. This may cause instability during training.")

if len(X) == 0:
    raise ValueError("No training samples remain after filtering. Please check your data and matching logic.")

# Encode labels
unique_labels = sorted(list(set(y)))
label_to_index = {l: i for i, l in enumerate(unique_labels)}
y_idx = np.array([label_to_index[l] for l in y])
y_cat = to_categorical(y_idx, num_classes=len(unique_labels))

# Add channel dimension
X = X[..., np.newaxis]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# 4. Build model
input_shape = (N_MELS + N_MFCC, TARGET_LENGTH, 1)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16
)

# 6. Save
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}") 