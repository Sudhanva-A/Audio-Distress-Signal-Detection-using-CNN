# Audio Distress Signal Detection 

A deep learning-based system to detect human distress signals like "Help", "Amma", "Kapadi", and "Aaah" from audio inputs of Kannada,English Languages using MFCC and Convolutional Neural Networks (CNNs). Includes real-time detection with microphone input.

## Features
- MFCC-based audio preprocessing
- CNN classifier trained on distress audio vs. background noise
- Real-time mic detection via `tkinter`
- Accuracy: 93%

## Tech Stack
- Python, TensorFlow/Keras
- Librosa, OpenCV
- Tkinter (for real-time GUI)
- Matplotlib, Scikit-learn

## Dataset
- WAV audio files of distress signals in English & Kannada
- Converted to MFCC spectrograms for training/testing

## How to Run
```bash
python real_time_classifier.py
