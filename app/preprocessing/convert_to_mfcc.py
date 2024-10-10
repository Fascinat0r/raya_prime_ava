import os

import librosa
import numpy as np

from app.utils.logger import get_logger  # Подключение логгера

# Инициализация логгера
logger = get_logger("mfcc_extraction")
logger.level = 0


def extract_mfcc(audio_file, output_file, n_mfcc=20, sr=16000, n_fft=2048, hop_length=512, fixed_length=157):
    """
    Извлечение MFCC признаков из аудиофайла и сохранение в виде .npy файла с гарантированной фиксированной длиной.

    :param audio_file: Путь к входному .wav файлу.
    :param output_file: Путь для сохранения выходного .npy файла.
    :param n_mfcc: Количество коэффициентов MFCC.
    :param sr: Частота дискретизации (по умолчанию 16 кГц).
    :param n_fft: Размер окна для FFT (по умолчанию 2048).
    :param hop_length: Шаг окон (по умолчанию 512).
    :param fixed_length: Фиксированное количество временных окон (целевое значение длины временного измерения).
    """
    logger.info(f"Загрузка аудиофайла: {audio_file}")

    # Загрузка аудиофайла с целевой частотой дискретизации
    y, sample_rate = librosa.load(audio_file, sr=sr)

    logger.info(f"Извлечение MFCC признаков для файла: {audio_file}")
    # Извлечение MFCC признаков
    mfcc_features = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Транспонирование, чтобы получить матрицу (время, n_mfcc)
    mfcc_features = mfcc_features.T

    # Логируем изначальный размер
    logger.debug(f"Изначальный размер MFCC: {mfcc_features.shape}")

    # Гарантирование фиксированной длины (дополнение или обрезка до фиксированной длины)
    if mfcc_features.shape[0] < fixed_length:
        # Если временная длина меньше целевой, дополняем нулями
        padding = np.zeros((fixed_length - mfcc_features.shape[0], n_mfcc))
        mfcc_features = np.vstack((mfcc_features, padding))
        logger.info(f"Файл {audio_file} дополнен нулями до {fixed_length} временных шагов.")
    elif mfcc_features.shape[0] > fixed_length:
        # Если временная длина больше целевой, обрезаем
        mfcc_features = mfcc_features[:fixed_length, :]
        logger.info(f"Файл {audio_file} обрезан до {fixed_length} временных шагов.")

    # Сохранение признаков в .npy файл
    np.save(output_file, mfcc_features)
    logger.info(f"MFCC признаки сохранены в {output_file} с размерностью {mfcc_features.shape}.")


# Пример использования:
if __name__ == "__main__":
    # Пример входного и выходного файла
    input_file = "../data/augmented/aug_0_Lp._Идеальный_МИР_#30_НОВЫЙ_ПЕРСОНАЖ_•_Майнкрафт_segment_1.wav"  # Путь к входному аудиофайлу
    output_file = "../data/mfcc/example_mfcc.npy"  # Путь для сохранения выходного .npy файла

    # Убедиться, что выходная папка существует
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Извлечение и сохранение MFCC с целевой длиной в 157 временных шагов
    extract_mfcc(input_file, output_file, fixed_length=157)
