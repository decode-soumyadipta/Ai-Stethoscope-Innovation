from flask import Flask, render_template, jsonify
import threading
import numpy as np
from real_time_processing import monitor_firebase, combine_audio, process_combined_audio, logs, prediction, combined_waveforms
from threading import Lock

app = Flask(__name__)

# Shared global variables for the dashboard
audio_waveform = []  # Holds raw audio data for real-time plotting
recent_predictions = []  # Store last 5 predictions
MAX_WAVEFORMS = 500  # Number of samples to retain for visualization
simplified_waveform = []  # Store the normalized, simplified graph data
SAMPLE_RATE = 16000

# Thread lock to ensure safe concurrent access
lock = Lock()


def normalize_waveform(waveform):
    """
    Normalize the waveform to lie within the range [-1, 1].
    """
    if len(waveform) == 0:
        return []
    max_val = max(abs(np.min(waveform)), np.max(waveform))
    return (waveform / max_val).tolist() if max_val != 0 else waveform.tolist()


def generate_simplified_waveform(waveform):
    """
    Generate a simplified, rhythmic waveform for the second graph.
    Simulates regular heartbeats using smoothing and normalization.
    """
    if len(waveform) == 0:
        return []

    # Down-sample for simplicity
    downsampled = waveform[::100]

    # Apply smoothing to mimic rhythmic peaks
    smoothed = np.convolve(downsampled, np.ones(30) / 30, mode='valid')
    normalized = normalize_waveform(smoothed)
    return normalized


def firebase_monitor():
    """
    Monitor Firebase in a separate thread to fetch real-time updates.
    """
    global audio_waveform, recent_predictions, simplified_waveform

    def handle_audio_data(audio_data):
        """
        Handles incoming audio data for both graphs and updates logs.
        """
        global audio_waveform, recent_predictions, simplified_waveform

        with lock:  # Ensure thread-safe updates
            # Combine and process audio
            if combine_audio(audio_data):
                print("2 seconds of audio combined. Processing...")
                logs.append("2 seconds of audio combined. Processing...")
                process_combined_audio()

                # Append prediction to recent predictions
                recent_predictions.append(prediction.copy())
                if len(recent_predictions) > 5:
                    recent_predictions.pop(0)

            # Update real-time graph data
            audio_waveform.extend(audio_data)
            if len(audio_waveform) > SAMPLE_RATE * MAX_WAVEFORMS:  # Retain data for visualization
                audio_waveform = audio_waveform[-(SAMPLE_RATE * MAX_WAVEFORMS):]

            # Update simplified graph
            simplified_waveform = generate_simplified_waveform(audio_waveform)

    # Start Firebase monitoring with the callback
    monitor_firebase(handle_audio_data)


@app.route('/')
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """
    Provide real-time data for the dashboard:
    - Real-time audio waveforms for graphing.
    - Simplified graph data.
    - Pipeline logs and predictions.
    """
    with lock:  # Ensure thread-safe access
        # Identify the most confident prediction
        most_confident_prediction = max(
            recent_predictions, key=lambda x: x["confidence"], default={"disease": "None", "confidence": 0.0}
        )

        # Prepare JSON-safe predictions
        safe_predictions = [{"disease": p["disease"], "confidence": float(p["confidence"])} for p in recent_predictions]
        most_confident_safe = {
            "disease": most_confident_prediction["disease"],
            "confidence": float(most_confident_prediction["confidence"])
        }

        # Normalize real-time waveform data
        normalized_waveform = normalize_waveform(audio_waveform)

        # Limit logs sent to the frontend for smoother rendering
        trimmed_logs = logs[-20:]  # Send only the last 20 logs

        return jsonify({
            "waveforms": [normalized_waveform],  # Real-time audio waveform
            "simplified_waveform": simplified_waveform,  # Simplified, rhythmic data
            "logs": trimmed_logs,  # Send trimmed logs to frontend
            "predictions": safe_predictions,  # JSON-safe predictions
            "most_confident_prediction": most_confident_safe
        })


if __name__ == "__main__":
    threading.Thread(target=firebase_monitor, daemon=True).start()
    app.run(debug=True, port=5000)
