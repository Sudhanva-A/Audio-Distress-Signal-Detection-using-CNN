import os
import numpy as np
import librosa
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import speech_recognition as sr


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

AUDIO_DIR = r'D:\FInal_year_project'
NOISE_DIR = r'D:\Background'
SAMPLE_RATE = 16000
DURATION = 2   
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40
MFCC_TIME_STEPS = 130

class_labels = ['Aaaaah', 'Amma', 'Kapadi', 'Help', 'Background']
label_map = {label: idx for idx, label in enumerate(class_labels)}


def load_audio(file_path):
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    elif len(signal) < SAMPLES_PER_TRACK:
        pad = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, pad))
    return signal

def extract_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    if mfcc.shape[0] < MFCC_TIME_STEPS:
        mfcc = np.pad(mfcc, ((0, MFCC_TIME_STEPS - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MFCC_TIME_STEPS, :]
    return mfcc

def apply_spec_augment(mfcc):
    max_time_mask = MFCC_TIME_STEPS // 10  
    max_freq_mask = N_MFCC // 10

    time_mask = random.randint(0, max_time_mask)
    time_start = random.randint(0, MFCC_TIME_STEPS - time_mask)
    mfcc[time_start:time_start+time_mask, :] = 0

    freq_mask = random.randint(0, max_freq_mask)
    freq_start = random.randint(0, N_MFCC - freq_mask)
    mfcc[:, freq_start:freq_start+freq_mask] = 0
    return mfcc

def augment_class_specific(signal, label):
    if label in ['Aaaaah', 'Amma']:
        signal = librosa.effects.pitch_shift(signal, sr=SAMPLE_RATE, n_steps=random.choice([-1, 1]))
    if label in ['Help', 'Kapadi']:
        signal = librosa.effects.time_stretch(signal, rate=random.uniform(0.9, 1.1))
    return signal

def chunk_background(signal, chunk_length=SAMPLES_PER_TRACK):
    chunks = []
    total_samples = len(signal)
    stride = SAMPLE_RATE   
    for start in range(0, total_samples - chunk_length + 1, stride):
        chunk = signal[start:start + chunk_length]
        if np.max(np.abs(chunk)) > 0.01: 
            chunks.append(chunk)
    return chunks


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError:
        return "Request error"


def load_amma_data(folder_path, label_map):
    X_amma, y_amma = [], []
    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(folder_path, file)
        signal = load_audio(file_path)

        
        if np.max(np.abs(signal)) < 0.01:
            continue

        mfcc = extract_mfcc(signal)
        if mfcc.shape != (MFCC_TIME_STEPS, N_MFCC):
            continue  
        X_amma.append(mfcc)
        y_amma.append(label_map['Amma'])
        
        augmented_signal = augment_class_specific(signal, 'Amma')
        mfcc_aug = extract_mfcc(augmented_signal)
        if mfcc_aug.shape != (MFCC_TIME_STEPS, N_MFCC):
            continue
        X_amma.append(mfcc_aug)
        y_amma.append(label_map['Amma'])

    return np.array(X_amma), np.array(y_amma)



print("Loading background noise...")
background_signals = []
for file in os.listdir(NOISE_DIR):
    if file.endswith(".wav"):
        path = os.path.join(NOISE_DIR, file)
        noise, _ = librosa.load(path, sr=SAMPLE_RATE)
        chunks = chunk_background(noise)
        background_signals.extend(chunks)

print(f"Extracted {len(background_signals)} background chunks")

X, y = [], []
print("Loading distress samples...")

for label in class_labels:
    if label == 'Background':
        continue
    folder = os.path.join(AUDIO_DIR, label)
    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue
        signal = load_audio(os.path.join(folder, file))

        mfcc = extract_mfcc(signal)
        mfcc = apply_spec_augment(mfcc)
        X.append(mfcc)
        y.append(label_map[label])

        augmented = augment_class_specific(signal, label)
        mfcc_aug = extract_mfcc(augmented)
        mfcc_aug = apply_spec_augment(mfcc_aug)
        X.append(mfcc_aug)
        y.append(label_map[label])


for chunk in background_signals:
    mfcc_bg = extract_mfcc(chunk)
    mfcc_bg = apply_spec_augment(mfcc_bg)
    X.append(mfcc_bg)
    y.append(label_map['Background'])


amma_folder_path = r'D:\FInal_year_project\Amma'  
X_amma, y_amma = load_amma_data(amma_folder_path, label_map)

X = np.concatenate((X, X_amma), axis=0)
y = np.concatenate((y, y_amma), axis=0)

X = np.array(X)[..., np.newaxis]
y = to_categorical(y, num_classes=5)

print(f"Final Dataset size: {X.shape[0]} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

y_train_labels = np.argmax(y_train, axis=1)
weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(weights))


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_model(X_train.shape[1:], len(class_labels))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, class_weight=class_weights
)
model.save(r"D:\FInal_year_project\distress_model.h5")



loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=class_labels))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(8,6))
sns.heatmap(cm * 100, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

class_wise_accuracy = cm.diagonal() * 100
plt.figure(figsize=(8,6))
sns.barplot(x=class_labels, y=class_wise_accuracy, palette='viridis')
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)
plt.title("Class-wise Accuracy (%)")
for i, v in enumerate(class_wise_accuracy):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)
plt.show()


y_test_bin = label_binarize(y_true, classes=[0,1,2,3,4])
y_pred_prob = model.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i, label in enumerate(class_labels):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
