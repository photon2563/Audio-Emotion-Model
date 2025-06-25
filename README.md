# 🎙️ Audio Emotion Recognition Web App using Deep Learning

This project focuses upon a deep learning model based upon **CNN + BiLSTM + MLP**, to detect the emotion of speaker. Google Colab has been strongly useed to prepare a notebook and train the model. In the model training, we have used RAVDESS speech and songs dataset. 

There are eight primary emotions being covered **(neutral, calm, happy, sad, angry, fearful, disgusted, surprised)**. In the model after loading, feature extraction has been done. A total of 9 features have been extracted, namely, **MFCC, Delta, Delta², Chroma, Mel, Spectral Contrast, Tonnetz, RMSC and ZCR.** 

After this, data augmentation has been done, only upon the training dataset. Three sort of augmentation including, **Pitch Shifting, Time Stretching and Gaussian Noise Addition**. This helps in robust training of the model. Further the data has been split into **4:1** for training and validation respectively with **stratification**. Afterwards **Label Encoding** is done with **one-hot** with scaling using **Standard Scalar**.

Following this, is the **CNN + BiLSTM + MLP** model, the roles of these three have been explained below. Further, Adam has been used with gradient clipping for optimization. **Categorical Crossentropy** loss function is used for multi-class classification. 

Then, model is trained, with patience = 8 along with callbacks with batch size = 32, 50 epochs are used and after a while, the validation accuracy became saturated around 82-84 % which is acceptable for the project. 

After this, the **confusion matrix**, and **accuracies(overall and class)** have been showed with **f1-score**.
Further, a web-app has been hosted using **Streamlit via GitHub**. The requirement file in the repo contains the necessary python dependencies for app hosting. 

The web-app URL is: https://audio-emotion-model-hxbawkp8gtdjeuazgfrjbs.streamlit.app/

NOTE: Python 3.9 has been used while hosting the app.


---

##  Project Summary

-  **Input**: `.wav` audio files (RAVDESS speech + song)
-  **Output**: Emotion class 
-  **Accuracy Target**:
  - Overall Accuracy: > 80%
  - F1 Score (Macro): > 80%
  - Class-wise Accuracy: > 75% (Except the sad class)
-  **Deployment**: Streamlit Web App hosted via GitHub

---

##  Code Architecture & Phases

###  Phase 1: Dataset & Label Preprocessing

- Two folders: `SPEECH/` and `SONGS/` from RAVDESS, each containing `.wav` files.
- Emotion labels are extracted from filenames using standard RAVDESS naming conventions.
- Files are labeled and encoded using `LabelEncoder`, and then split into 80% training and 20% validation (for both song and speech data).

---

###  Phase 2: Feature Extraction (with Functional Descriptions)

We extract a **comprehensive set of audio features** from each audio clip to help the model understand tone, rhythm, pitch, and energy dynamics.

| Feature | Description |
|--------|-------------|
| **MFCCs** | Capture timbral texture of speech; mimic human ear perception of frequencies. |
| **Delta & Delta²** | Represent speed and acceleration of MFCC changes; highlight tempo/emotion transitions. |
| **Chroma STFT** | Capture pitch class and harmonics — useful for identifying melody, especially in songs. |
| **Mel Spectrogram** | Time-frequency representation with perceptual Mel scale; identifies energy distribution across frequencies. |
| **Spectral Contrast** | Measures contrast between spectral peaks and valleys; highlights emotional intensity differences. |
| **Tonnetz** | Harmonic representation; helpful in identifying consonance/dissonance — useful for calm/sadness. |
| **RMSE** | Captures loudness and energy dynamics over time; correlates with emotional force. |
| **ZCR** | Measures frequency of signal crossings; high in tense or sharp emotional speech (anger, fear). |

All features are stacked and shaped into a fixed-length 2D array (using padding or cropping), then scaled using `StandardScaler`.

---

###  Phase 3: Augmentation (Training Only)

To improve generalization and prevent overfitting:

-  Pitch Shifting
-  Time Stretching
-  Gaussian Noise Addition

These are applied **only to the training dataset**, helping the model generalize better to real-world variations.

---

###  Phase 4: Model Architecture – CNN + BiLSTM + MLP

####  Overview
Our model is a hybrid of **Convolutional layers** (to learn spatial patterns), **Bidirectional LSTM layers** (to learn temporal sequences), and **Dense MLP layers** (for final classification).

In the model architecture, sequential CNN and BiLSTM have been used. CNN layers (1-D) have been used, with kernel size = 3, with batch normalization and **'relu'** aactivation, making it deep. Following, we have have two BiLSTM layers and MLP with **relu** and **Softmax** activation.
The CNN models have 64, 128 and 128 filters/hyperparameters respectively followed by BiLSTM's 128 and 64.

####  Layer Functions:

| Layer | Role |
|-------|------|
| **CNN** | Detects local frequency-time features (e.g., pitch rises, rhythm changes). Great for pattern recognition in spectrogram-like data. |
| **BiLSTM** | Captures long-term dependencies and sequence context **in both directions** — future and past. Vital for modeling how emotion evolves over time. |
| **MLP (Dense Layers)** | Final classification layer; applies learned weights and outputs emotion probabilities. Includes dropout and label smoothing to prevent overfitting. 

**Other Enhancements**:
-  **Batch Normalization**
-  **Dropout Layers**
-  **Gradient Clipping**
-  **Label Smoothing** (added to categorical cross-entropy loss)
-  **EarlyStopping & ModelCheckpoint** for training control

---

###  Phase 5: Evaluation & Metrics

We evaluate the model on validation data using:

-  **Accuracy**
-  **F1 Score (macro)**
-  **Per-class Accuracy**
-  **Confusion Matrix Heatmap** using seaborn

All trained components are saved:
- `best_model.h5`
- `scaler.pkl`
- `label_encoder.pkl`

---

###  Phase 6: Streamlit Web App

An easy-to-use web app built with Streamlit:
- Upload a `.wav` file
- Features are extracted and scaled live
- The model classifies the emotion
- Confidence for each emotion is displayed in a neat output

---

##  Model Performance Snapshot
### Actual Snapshot of the accuracy table after model training.
 * Overall Accuracy: 83.61%
 
 * Macro F1 Score: 83.73%
 
  Classification Report:
               precision      recall     f1-score     support
               
       angry     0.9853    0.8933    0.9371        75
     
        calm     0.8684    0.8919    0.8800        74
     
     disgust     0.9143    0.8649    0.8889        37
     
     fearful     0.7326    0.8514    0.7875        74
     
       happy     0.8472    0.8243    0.8356        74
     
     neutral     0.6977    0.8108    0.7500        37
     
         sad     0.7794    0.7162    0.7465        74
     
   surprised     0.9118    0.8378    0.8732        37


    accuracy                         0.8361       482
   macro avg     0.8421    0.8363    0.8373       482
weighted avg     0.8426    0.8361    0.8375       482

Class-wise Accuracy:
* angry      : 89.33%
* calm       : 89.19%
* disgust    : 86.49%
* fearful    : 85.14%
* happy      : 82.43%
* neutral    : 81.08%
* **sad        : 71.62%**
* surprised  : 83.78%

| Metric            | Target Achieved |
|-------------------|------------------|
| Overall Accuracy  |  > 80% |
| F1 Score (macro)  |  > 80% |
| Per-Class Accuracy|  > 75% (except Sad) |
| Confusion Matrix  |  Visualized |

**Sad emotional class accuracy stood low because of these probable reasons:**
* Acoustic Similarity to Other Emotions like neutral, calm resembling low pitch and slow tempo features.
* Features like ZCR, RMSE are not able to differentiate sad tone because of flat, stable, low-energy profiles.
* Another reason could be pitch shifting or time stretching sad audio might distort its already-subtle characteristics.

