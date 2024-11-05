# app/preprocessing/find_silence.py
# Description: Находит интервалы тишины в аудиофайле.

import librosa
import pandas as pd


def detect_silence_intervals(audio_path, silence_threshold=-25, min_silence_duration=0.05):
    """
    Находит интервалы тишины в аудиофайле.

    :param audio_path: Путь к аудиофайлу.
    :param silence_threshold: Порог тишины в децибелах (относительно полной шкалы).
    :param min_silence_duration: Минимальная продолжительность тишины для учета (в секундах).
    :return: DataFrame с интервалами тишины.
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Находим интервалы с голосом, используя заданный порог
    voice_intervals = librosa.effects.split(y, top_db=-silence_threshold, hop_length=int(sr * min_silence_duration))

    # Создаем список интервалов тишины как промежутков между интервалами с голосом
    silence_intervals = []
    previous_end = 0  # Начало аудио

    for start, end in voice_intervals:
        if start > previous_end:
            silence_intervals.append((previous_end, start))  # Интервал тишины перед текущим интервалом с голосом
        previous_end = end

    # Добавляем последний интервал тишины, если он есть, до конца аудио
    if previous_end < len(y):
        silence_intervals.append((previous_end, len(y)))

    # Преобразуем интервалы в DataFrame с указанием времени
    silence_df = pd.DataFrame(silence_intervals, columns=['start_frame', 'end_frame'])
    silence_df['start_time'] = silence_df['start_frame'] / sr
    silence_df['end_time'] = silence_df['end_frame'] / sr
    silence_df = silence_df[['start_time', 'end_time']]
    return silence_df
