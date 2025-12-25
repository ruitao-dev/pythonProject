import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import soundfile as sf
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with open(file_path, 'rb') as f:
        audio, sample_rate = librosa.load(f, sr=None)
    features = []
    if mfcc:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_mean)
    if chroma:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        features.extend(chroma_mean)
    if mel:
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_mean = np.mean(mel.T, axis=0)
        features.extend(mel_mean)
    return np.array(features)
def visualize_audio(audio_path):
    audio, sr = librosa.load(audio_path)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f  dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
def extract_segment(audio_path, start_time, end_time, output_path):
    audio, sr = librosa.load(audio_path, sr=None)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]
    sf.write(output_path, segment, sr)
    print(f"Segment saved to {output_path}")
    return segment
def train_cough_classifier(cough_files, non_cough_files):
    X = []
    y = []
    for file in cough_files:
        features = extract_features(file)
        X.append(features)
        y.append(1)  
    for file in non_cough_files:
        features = extract_features(file)
        X.append(features)
        y.append(0)  
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    return model
if __name__ == "__main__":
    cough_audio="685484/C16F2448000022_L31.3.1_20250723220901_AB002411C_FAR.ogg"  
   # non_cough_audio = "AudioSetWav16k/eval_segments/-1nilez17Dg_30.000.wav"
   # print("Visualizing cough audio:")
   # visualize_audio(cough_audio)
    print("\nExtracting cough segment:")
    cough_segment = extract_segment(cough_audio,1758.0,1760.0,"C16F2448000022_L31.3.1_20250723220901_AB002411C_FAR_1758.ogg")
   # print("\nTraining cough classifier:")
   # model = train_cough_classifier([cough_audio], [non_cough_audio])
   # test_audio = "AudioSetWav16k/eval_segments/-22tna7KHzI_28.000.wav"
   # if os.path.exists(test_audio):
   #    features = extract_features(test_audio)
   #    prediction = model.predict([features])
   #    print(f"\nPrediction for {test_audio}: {'Cough' if prediction[0] == 1 else 'Not Cough'}")
