# app/predict/split_audio.py
# Description: Модуль для разделения аудиофайла на два: куски, находящиеся внутри и вне указанных интервалов.
import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def split_audio_by_intervals(audio_path, intervals_path, target_output="target_audio.mp3",
                             non_target_output="non_target_audio.mp3"):
    """
    Делит аудиофайл на два: куски, находящиеся внутри и вне указанных интервалов.

    :param audio_path: Путь к исходному аудиофайлу в формате WAV.
    :param intervals_path: Путь к CSV-файлу с интервалами (столбцы 'start_time' и 'end_time').
    :param target_output: Путь для сохранения аудио внутри интервалов.
    :param non_target_output: Путь для сохранения аудио вне интервалов.
    """
    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=None)

    # Загрузка интервалов
    intervals = pd.read_csv(intervals_path)

    # Преобразование времени в отсчеты (фреймы)
    target_audio = []
    non_target_audio = []
    last_end = 0

    for _, row in intervals.iterrows():
        start_sample = int(row['start_time'] * sr)
        end_sample = int(row['end_time'] * sr)

        # Добавляем кусок вне интервала в нецелевое аудио
        if last_end < start_sample:
            non_target_audio.append(y[last_end:start_sample])

        # Добавляем кусок внутри интервала в целевое аудио
        target_audio.append(y[start_sample:end_sample])
        last_end = end_sample

    # Добавляем оставшуюся часть вне интервалов в нецелевое аудио
    if last_end < len(y):
        non_target_audio.append(y[last_end:])

    # Объединение и сохранение сегментов как отдельных аудиофайлов
    target_audio = np.concatenate(target_audio) if target_audio else np.array([])
    non_target_audio = np.concatenate(non_target_audio) if non_target_audio else np.array([])

    # Сохранение файлов
    if target_audio.size > 0:
        sf.write(target_output, target_audio, sr)
        print(f"Целевое аудио сохранено в '{target_output}'")
    else:
        print("Целевое аудио отсутствует.")

    if non_target_audio.size > 0:
        sf.write(non_target_output, non_target_audio, sr)
        print(f"Нецелевое аудио сохранено в '{non_target_output}'")
    else:
        print("Нецелевое аудио отсутствует.")


# Пример использования
if __name__ == "__main__":
    split_audio_by_intervals("data/output_segment.wav", "data/processed_intervals.csv")
