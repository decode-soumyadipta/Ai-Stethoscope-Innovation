
#  **Stethogen AI Stethoscope: MFCC-Enhanced Tele-Auscultation Platform** 

 🧑‍⚕️ Next-Gen AI Stethoscope with MFCC & Mel-Spectrogram Fusion for **✅Real-Time**, ✅**Accurate**, **✅Affordable**, Cloud-Connected Diagnostics

---

> _“Giving AI a doctor’s ear—one MFCC at a time.”_ 

---
<br>

🔴 Product - Advanced development stage (PCB  designing, improving accuracy, extensive data collection & training, fine-tuning)

## 🌟 Overview

**Stethogen** is an innovative, AI-powered stethoscope platform that leverages advanced audio signal processing and deep learning to revolutionize heart and lung sound diagnostics. Unlike traditional digital stethoscopes, Stethogen fuses **Mel-Frequency Cepstral Coefficients (MFCCs)** and **Mel-spectrograms** for robust, human-auditory-inspired disease detection—setting a new benchmark in telemedicine and smart healthcare devices.

![DFVFE](https://github.com/user-attachments/assets/0ac587c5-dc53-4b2e-895a-e467ce7656f2)

![image](https://github.com/user-attachments/assets/e2b8de29-f718-45ff-8f7b-772a92fa71cd)

---

## 🧠 Core Innovations

- **MFCC + Mel-Spectrogram Fusion:**  
  Simultaneously extracts MFCCs and Mel-spectrograms from auscultation audio, stacking them as model input for superior feature richness and diagnostic accuracy.  
  _MFCCs are the gold standard in speech and biomedical audio analysis, rarely used in real-time stethoscope dashboards—this is a true leap beyond the status quo!_

- **Real-Time Deep Learning:**  
  Processes live audio from an ESP32-based stethoscope, predicts diseases (e.g., Asthma, COPD, Heart Failure, Lung Fibrosis, etc.), and displays confidence scores instantly.

- **Interactive Dashboard:**  
  - **Live Waveform & Heartbeat Graphs:** Visualize raw and processed audio in real time.
  - **MFCC Visualization:** See the MFCC heatmap for each new recording—showcasing the AI’s “ear.”
  - **Disease Prediction & Confidence Meter:** Get instant, interpretable results.
  - **Pipeline Logs:** Transparent, step-by-step processing feedback.

- **Cloud Integration:**  
  All data and predictions are synced with Firebase for secure, scalable storage and remote access.

---

![stethogen final](https://github.com/user-attachments/assets/b7da0f5e-e74f-41f3-8eb0-f2bb487c1fe0)

![kfhk](https://github.com/user-attachments/assets/c14ebf9d-27ea-4a1c-8a01-bb9b65a66e2f)

### Initial Model Training:

![image](https://github.com/user-attachments/assets/ecba9704-8556-43e5-bfaa-f8c52165531c)

<br>

### 🀄Dataset Link (Kaggle): https://www.kaggle.com/datasets/arashnic/lung-dataset

<br>
---


## 🚀 Unique Features & Differentiators

- **MFCC-Driven AI:**  
  Most digital stethoscopes use only basic features or raw waveforms. StethoAI’s MFCC+Mel approach is inspired by how the human ear and brain process sound, making it more robust to noise and variability.

- **Plug-and-Play Telemedicine Ready:**  
  Designed for future expansion:  
  _A single click will soon send live lung/heart sound graphs, audio, and AI predictions to a doctor’s dashboard for instant tele-consultation._ 🩺💻

- **Device Scalability:**  
  Built-in (and future) support for multiple device IDs—enabling hospital-wide or multi-patient deployments.

- **Open, Modular, and Extensible:**  
  Clean Python/Flask codebase, easy to extend for new diseases, sensors, or cloud platforms.

---

## 🗝️ Key Features

- ESP32-based I2S MEMS microphone for high-fidelity sound capture
- Real-time waveform and MFCC visualization
- Deep learning model trained on both MFCC and Mel-spectrogram features
- Disease prediction with confidence scoring
- Firebase cloud sync
- Modern, responsive dashboard (Flask + Chart.js)
- Device ID input for future multi-device support
- Robust `.gitignore` and modular code structure

---

## 🏥 Future Vision: Telemedicine & Remote Care

> **Imagine:**  
> A patient records their lung sounds at home. With one click, the live waveform, MFCC graph, and AI prediction are sent to a doctor’s dashboard—enabling instant, expert tele-consultation, even in remote areas.  
> _Stethogen is built for this future._

---

## 📂 Project Structure

```
ai-stethoscope/
├── dashboard/
│   ├── app.py                # Flask dashboard backend
│   ├── real_time_processing.py # Real-time audio processing & prediction
│   ├── static/               # Static files (JS, CSS, MFCC images)
│   └── templates/            # HTML templates
├── datasets/                 # Audio and annotation data
├── models/                   # Trained models (excluded from git)
├── train_model.py            # Model training script (MFCC+Mel)
├── requirements.txt          # Python dependencies
├── .gitignore                # Robust ignore rules
└── README.md                 # This file!
```

---

## ⚡ Quick Start

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```sh
   python train_model.py
   ```

3. **Run the dashboard:**
   ```sh
   python dashboard/app.py
   ```
   Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🤝 Contributing & Contact

- Contributions, ideas, and collaborations are welcome!
- For questions or demo requests, contact [soumyadiptadey7@gmail.com].

---

## 🏆 Why Stethogen is Different

- **MFCC+Mel fusion** for best-in-class audio diagnostics
- **Telemedicine-ready** architecture
- **Transparent, real-time AI** for both patients and clinicians
- **Open, extensible, and future-proof**



**[GitHub Repo](https://github.com/decode-soumyadipta/Ai-Stethoscope-Innovation)**

