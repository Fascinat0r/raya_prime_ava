import librosa
import pickle
import numpy as np
import os


# Извлечение MFCC и дополнительных признаков
def extract_mfcc(audio_data):
    y, sr = audio_data
    # Извлечение MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc


# Сохранение MFCC в файл
def save_mfcc(mfcc, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(mfcc, f)


# Загрузка MFCC из файла
def load_mfcc(load_path):
    with open(load_path, 'rb') as f:
        mfcc = pickle.load(f)
    return mfcc
