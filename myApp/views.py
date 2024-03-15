from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json

import numpy as np
import tensorflow as tf
import librosa
import pickle
from keras.models import load_model
# from sklearn.preprocessing import StandardScaler
global graph,model

tf.config.run_functions_eagerly(True)

#initializing the graph
graph = tf.compat.v1.get_default_graph()

#loading our trained model
print("Keras model loading.......")
model = load_model('myApp/finalModel.h5')
print("Model loaded!!")


# Create your views here.
def hello(request):
    return HttpResponse('hello!')



def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Split audio into 3-second frames
    frame_length = 3 * sr  # 3 seconds
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length)

    all_features = []

    for frame in frames.T:
        # Extract features for each frame
        chroma_stft = librosa.feature.chroma_stft(y=frame, sr=sr)
        rms = librosa.feature.rms(y=frame)
        spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=frame)
        harmony = librosa.effects.harmonic(y=frame)
        perceptr = librosa.feature.spectral_flatness(y=frame)
        tempo, _ = librosa.beat.beat_track(y=frame, sr=sr)
        mfccs = librosa.feature.mfcc(y=frame, sr=sr)

        # Compute mean and variance for each feature
        features = []
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))
        features.append(np.mean(rms))
        features.append(np.var(rms))
        features.append(np.mean(spectral_centroid))
        features.append(np.var(spectral_centroid))
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))
        features.append(np.mean(spectral_rolloff))
        features.append(np.var(spectral_rolloff))
        features.append(np.mean(zero_crossing_rate))
        features.append(np.var(zero_crossing_rate))
        features.append(np.mean(harmony))
        features.append(np.var(harmony))
        features.append(np.mean(perceptr))
        features.append(np.var(perceptr))
        features.append(tempo)

        # Compute mean and variance for MFCCs
        for i in range(1, 21):
            features.append(np.mean(mfccs[i-1]))
            features.append(np.var(mfccs[i-1])) 
        

        all_features.append(features)

    return all_features

def get_pred(feature):
    genre = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    feature = np.array(feature)
    examp = feature.reshape(1, -1)
    with open('myApp/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    examp = scaler.transform(examp)
    preds = model.predict(examp)
    preds = preds.flatten()
    gen = genre[np.where(preds==max(preds))[0][0]]
    return gen


def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        genre = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        myfile = request.FILES['myfile']

        arr = []
        features = extract_features(myfile)
        for feature in features:
            arr.append(get_pred(feature))
        
        ans= max(set(arr), key=arr.count)
        return JsonResponse({'prediction': ans})