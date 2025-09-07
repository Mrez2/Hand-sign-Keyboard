# ✋ Hand Sign Keyboard  
<img width="845" height="879" alt="Screenshot 2025-09-07 182547" src="https://github.com/user-attachments/assets/f896e4e5-5986-431f-a483-c8868ea8d102" />

A real-time hand sign recognition system using **PyTorch**, **Mediapipe**, and **OpenCV**.  
This project lets you **train models to recognize custom gestures** and map them to letters or words, enabling gesture-based typing and accessibility tools for deaf and hard-of-hearing communities.  

---

## 🚀 Features  
- Real-time hand tracking with Mediapipe  
- Trainable PyTorch model for sign recognition  
- Map gestures to letters, words, or actions  
- Build inclusive tools for accessibility and communication  

---

## 🛠️ Tech Stack  
- [PyTorch](https://pytorch.org/) – Deep Learning framework  
- [Mediapipe](https://developers.google.com/mediapipe) – Hand landmark detection  
- [OpenCV](https://opencv.org/) – Visualization & camera handling  

---

## 📂 Project Structure  
Hand-sign-Keyboard/
│── collect_data.py # Collect hand gesture samples
│── train.py # Train PyTorch model on collected data
│── infer_realtime.py # Real-time gesture recognition & visualization
│── data/raw/ # Saved hand sign datasets (.npy files)
│── models/ # Trained models (best_model.pt)

yaml
Copy code

---

## ⚡ Installation  
Clone the repo and install dependencies:  
```bash
git clone https://github.com/Mrez2/Hand-sign-Keyboard.git
cd Hand-sign-Keyboard
pip install -r requirements.txt
▶️ Usage
1️⃣ Collect Data
Run webcam data collection and press letter keys to save samples:

bash
Copy code
python collect_data.py
2️⃣ Train Model
Train the PyTorch model on your collected data:

bash
Copy code
python train.py
3️⃣ Run Real-Time Inference
Use your webcam to recognize gestures live:

bash
Copy code
python infer_realtime.py
🌍 Applications
Gesture-based typing

Accessibility tools for deaf/hard-of-hearing users

Human-computer interaction without keyboard/touchscreen
