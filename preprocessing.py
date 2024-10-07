import os
import librosa
import numpy as np

def load_audio(file_path, sr=16000):
    """Загружает аудио и возвращает сигнал и частоту дискретизации."""
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def extract_mfcc(audio, sample_rate, n_mfcc=13):
    """Извлекает MFCC из аудиосигнала."""
    return librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T

def normalize_audio(audio):
    """Приводит аудио к стандартной частоте дискретизации и убирает шумы."""
    return librosa.effects.preemphasis(audio)
