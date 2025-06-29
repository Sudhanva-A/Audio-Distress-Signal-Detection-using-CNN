import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model


SAMPLE_RATE = 16000
DURATION = 2
N_MFCC = 40
MFCC_TIME_STEPS = 130
MODEL_PATH = r'D:\FInal_year_project\distress_model.h5'  

class_labels = ['Aaaaah', 'Amma', 'Kapadi', 'Help', 'Background']


model = load_model(MODEL_PATH)


def extract_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MFCC_TIME_STEPS:
        mfcc = np.pad(mfcc, ((0, MFCC_TIME_STEPS - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MFCC_TIME_STEPS, :]
    return mfcc[..., np.newaxis]  

def record_audio(duration=2, fs=16000):
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        print("Recording complete.")
        return recording.flatten()
    except Exception as e:
        print(f"Error recording: {e}")
        return None


def classify_audio():
    signal = record_audio(duration=DURATION, fs=SAMPLE_RATE)
    if signal is None:
        return

    mfcc = extract_mfcc(signal)
    mfcc = np.expand_dims(mfcc, axis=0)  

    predictions = model.predict(mfcc)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_class = class_labels[predicted_idx]

    if predicted_class != 'Background':
        status_msg = classify_confidence(confidence, predicted_class)
        status_label.config(text=status_msg, fg=status_color(confidence))
        result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f})")
    else:
        result_label.config(text=f"Prediction: {predicted_class}")
        status_label.config(text="No distress detected", fg='green')


def classify_confidence(conf, label):
    if conf <= 0.99:
        return f"No stress detected"
    else:
        return f"Stress Detected!!({label})"

def status_color(conf):
    if conf <= 0.99:
        return 'green'

    else:
        return 'red'


window = tk.Tk()
window.title("Real-Time Distress Sound Detector")
window.geometry("400x250")

title = tk.Label(window, text="Distress Sound Classifier", font=("Arial", 16))
title.pack(pady=10)

start_btn = tk.Button(window, text="Start Recording", font=("Arial", 12), command=classify_audio)
start_btn.pack(pady=10)

result_label = tk.Label(window, text="Prediction: N/A", font=("Arial", 12))
result_label.pack(pady=10)

status_label = tk.Label(window, text="", font=("Arial", 12, 'bold'))
status_label.pack(pady=5)

quit_btn = tk.Button(window, text="Exit", command=window.destroy)
quit_btn.pack(pady=10)

window.mainloop()
