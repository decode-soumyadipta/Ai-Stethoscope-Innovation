import requests
import base64
import os

FIREBASE_URL = "https://stethogen-default-rtdb.asia-southeast1.firebasedatabase.app/audio.json"
OUTPUT_AUDIO_DIR = "cloud/audio_files"

# Ensure output directory exists
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

def fetch_latest_audio():
    """Fetch the latest audio recording from Firebase."""
    response = requests.get(FIREBASE_URL)
    if response.status_code == 200:
        data = response.json()
        if data:
            latest_key = max(data.keys())
            return data[latest_key], latest_key
        else:
            print("No audio data found")
            return None, None
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None, None

def decode_base64_audio(encoded_audio, output_file="audio.wav"):
    """Decode Base64 encoded audio data and save to a WAV file."""
    audio_data = base64.b64decode(encoded_audio)
    output_path = os.path.join(OUTPUT_AUDIO_DIR, output_file)
    with open(output_path, "wb") as f:
        f.write(audio_data)
    print(f"Audio file saved: {output_path}")
    return output_path

if __name__ == "__main__":
    audio_data, audio_key = fetch_latest_audio()
    if audio_data:
        decode_base64_audio(audio_data, f"{audio_key}.wav")
