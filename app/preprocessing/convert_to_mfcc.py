import os

import librosa
import numpy as np


def extract_mfcc(audio_file, output_file, n_mfcc=20, sr=16000, n_fft=2048, hop_length=512):
    """
    Извлечение MFCC признаков из аудиофайла и сохранение в виде .npy файла.

    :param audio_file: Путь к входному .wav файлу.
    :param output_file: Путь для сохранения выходного .npy файла.
    :param n_mfcc: Количество коэффициентов MFCC.
    :param sr: Частота дискретизации (по умолчанию 16 кГц).
    :param n_fft: Размер окна для FFT (по умолчанию 2048).
    :param hop_length: Шаг окон (по умолчанию 512).
    """
    # Загрузка аудиофайла
    y, sample_rate = librosa.load(audio_file, sr=sr)

    # Извлечение MFCC признаков
    mfcc_features = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Транспонирование для сохранения (время x n_mfcc)
    mfcc_features = mfcc_features.T

    # Сохранение признаков в .npy файл
    np.save(output_file, mfcc_features)
    print(f"MFCC признаки сохранены в {output_file}.")


# Пример использования:
if __name__ == "__main__":
    input_file = "../data/augmented/target_augmented_1245.wav"  # Путь к входному аудиофайлу
    output_file = "../data/mfcc/example_mfcc.npy"  # Путь для сохранения выходного .npy файла

    # Убедиться, что выходная папка существует
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Извлечение и сохранение MFCC
    extract_mfcc(input_file, output_file)
