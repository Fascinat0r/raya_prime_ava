import os

import numpy as np
import torchaudio
from torchaudio.transforms import MFCC

from app.utils.logger import get_logger  # Подключение логгера

# Инициализация логгера
logger = get_logger("mfcc_extraction")


def extract_mfcc(audio_file, output_file, n_mfcc=20, sr=16000, n_fft=400, hop_length=160, fixed_length=157):
    """
    Извлечение MFCC признаков из аудиофайла с использованием torchaudio и сохранение
    в виде .npy файла с гарантированной фиксированной длиной.

    :param audio_file: Путь к входному .wav файлу.
    :param output_file: Путь для сохранения выходного .npy файла.
    :param n_mfcc: Количество коэффициентов MFCC.
    :param sr: Частота дискретизации (по умолчанию 16 кГц).
    :param n_fft: Размер окна для FFT.
    :param hop_length: Шаг окон.
    :param fixed_length: Фиксированное количество временных окон (целевое значение длины временного измерения).
    """
    logger.info(f"Загрузка аудиофайла: {audio_file}")

    # Загрузка аудиофайла с использованием torchaudio
    waveform, sample_rate = torchaudio.load(audio_file)

    # Приведение частоты дискретизации к целевой, если отличается
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
        sample_rate = sr

    logger.info(f"Извлечение MFCC признаков для файла: {audio_file} с частотой дискретизации {sample_rate} Гц")

    # Инициализация преобразования для извлечения MFCC
    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mfcc,
            "center": False
        }
    )

    # Извлечение MFCC признаков
    mfcc_features = mfcc_transform(waveform).squeeze(0).T  # Получаем матрицу размером (временные шаги, n_mfcc)

    # Преобразование MFCC признаков в numpy и логирование размеров
    mfcc_features = mfcc_features.numpy()
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
    else:
        logger.info(f"Файл {audio_file} уже имеет целевую длину в {fixed_length} временных шагов.")

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
