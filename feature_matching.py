import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import os

# Загрузка предобученной модели для распознавания голоса
MODEL_PATH = os.path.join(os.path.dirname(__file__), "voice_recognition_model.h5")
model = load_model(MODEL_PATH)


def extract_voice_features(mfcc):
    """Использует нейросеть для извлечения голосовых признаков."""
    return model.predict(np.expand_dims(mfcc, axis=0))[0]


def calculate_similarity(reference_features, sample_features):
    """Вычисляет схожесть между эталонными и текущими признаками."""
    return cosine_similarity([reference_features], [sample_features])[0][0]


def match_segments(reference_features, audio_file, threshold=0.75, segment_duration=5):
    """Ищет совпадения эталонного голоса в аудиофайле и возвращает таймкоды совпадений."""
    from preprocessing import load_audio, extract_mfcc, normalize_audio

    audio, sr = load_audio(audio_file)
    audio = normalize_audio(audio)
    step = int(segment_duration * sr)
    matches = []
    for start in range(0, len(audio) - step, step // 2):
        segment = audio[start:start + step]
        mfcc = extract_mfcc(segment, sr)
        sample_features = extract_voice_features(mfcc)
        similarity = calculate_similarity(reference_features, sample_features)
        if similarity >= threshold:
            matches.append((start / sr, (start + step) / sr, similarity))
    return matches
