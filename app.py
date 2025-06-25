import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os

# --------------------------------------------
# 📦 Load model and preprocessing tools
# --------------------------------------------
MODEL_PATH = "best_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# --------------------------------------------
# 🎧 Feature Extraction Function
# --------------------------------------------
def extract_features(audio_path, max_len=173):
    X, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    mel = librosa.feature.melspectrogram(y=X, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=X, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr)
    rmse = librosa.feature.rms(y=X)
    zcr = librosa.feature.zero_crossing_rate(y=X)

    # Combine all features
    features = np.vstack([mfcc, delta, delta2, chroma, mel, contrast, tonnetz, rmse, zcr])

    # Pad or crop to fixed length
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]

    return features.T  # shape: (time_steps, features)

# --------------------------------------------
# 🌐 Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="Audio Emotion Classifier", layout="centered")
st.title("🎙️ Audio Emotion Classifier")
st.markdown("Upload a `.wav` file to detect the emotion.")

uploaded_file = st.file_uploader("Choose a `.wav` audio file", type=["wav"])

if uploaded_file is not None:
    # Save temporary file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract & scale features
    features = extract_features("temp.wav")
    features_scaled = scaler.transform(features).reshape(1, features.shape[0], features.shape[1])

    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    # Show result
    st.success(f"🧠 Predicted Emotion: **{predicted_label[0].capitalize()}**")

    # Show probabilities
    st.subheader("📊 Prediction Confidence:")
    for label, prob in zip(label_encoder.classes_, prediction[0]):
        st.write(f"**{label.capitalize()}**: {prob:.2%}")
