import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from train_model import preprocess_audio, generate_spectrogram

# Load the trained model
MODEL_PATH = "models/model.h5"
model = load_model(MODEL_PATH)

# Labels used during training
LABELS = ['asthma', 'bron', 'copd', 'heart failure', 'lung fibrosis', 
          'n', 'plueral effusion', 'pneumonia', 
          'asthma and lung fibrosis', 'heart failure + lung fibrosis', 'heart failure + copd']

def predict(audio_file):
    """
    Predict the condition based on the input audio file.
    """
    # Preprocess the audio file
    audio, sr = preprocess_audio(audio_file)

    # Generate spectrogram
    spectrogram = generate_spectrogram(audio, sr)

    # Reshape the spectrogram for the model
    input_data = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(input_data)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Map index to label
    predicted_class = LABELS[predicted_class_idx]

    return predicted_class, confidence

if __name__ == "__main__":
    test_file = "cloud/audio_files/test_audio.wav"
    predicted_class, confidence = predict(test_file)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
