import random

import librosa
import numpy as np
from joblib import Parallel, delayed

from utils.my_logger import logger

# Параметры для параллельной обработки
NUM_CORES = -1  # Использование всех доступных ядер


def augment_audio(segments, target_length):
    """
    Увеличение количества сегментов путем дублирования и применения изменений с параллельной обработкой.
    :param segments: Список исходных аудио-сегментов
    :param target_length: Необходимое количество сегментов
    :return: Список сегментов с увеличением количества
    """
    logger.info(
        f"Начало аугментации аудио-сегментов. Исходное количество: {len(segments)}, целевое количество: {target_length}")
    augmented_segments = segments.copy()

    # Использование параллельной обработки для добавления аугментированных сегментов
    while len(augmented_segments) < target_length:
        remaining = target_length - len(augmented_segments)
        logger.info(f"Параллельное выполнение аугментации для {remaining} сегментов.")

        # Параллельное выполнение аугментаций
        new_segments = Parallel(n_jobs=NUM_CORES)(
            delayed(random_augmentation)(segment) for segment in random.choices(segments, k=remaining)
        )
        augmented_segments.extend(new_segments)
        logger.debug(
            f"Добавлены новые сегменты после аугментации. Текущее количество сегментов: {len(augmented_segments)}")

    logger.info(f"Завершена аугментация. Итоговое количество сегментов: {len(augmented_segments)}")
    return augmented_segments[:target_length]


def random_augmentation(segment):
    """
    Применение случайной аугментации к аудиосегменту.
    :param segment: Входной аудиосегмент в виде массива numpy
    :return: Модифицированный сегмент
    """
    augmentations = [
        change_pitch(segment, n_steps=random.choice([-2, -1, 1, 2])),  # Изменение тона
        change_speed(segment, rate=random.choice([0.9, 1.1, 1.2])),  # Изменение скорости
        add_noise(segment, noise_factor=0.005)  # Добавление шума
    ]
    return random.choice(augmentations)


def change_pitch(data, sr=16000, n_steps=2):
    """
    Изменение тона аудиофайла.
    :param data: Входной аудиофайл в виде массива numpy
    :param sr: Частота дискретизации
    :param n_steps: Количество шагов изменения тона (положительное или отрицательное значение)
    :return: Аудиофайл с измененным тоном
    """
    logger.debug(f"Изменение тона на {n_steps} шагов (частота дискретизации: {sr})")
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)


def change_speed(data, rate=1.0):
    """
    Изменение скорости воспроизведения аудиофайла.
    :param data: Входной аудиофайл в виде массива numpy
    :param rate: Коэффициент изменения скорости (меньше 1 — медленнее, больше 1 — быстрее)
    :return: Аудиофайл с измененной скоростью
    """
    logger.debug(f"Изменение скорости на коэффициент {rate}.")
    return librosa.effects.time_stretch(y=data, rate=rate)


def add_noise(data, noise_factor=0.005):
    """
    Добавление шума к аудиофайлу.
    :param data: Входной аудиофайл в виде массива numpy
    :param noise_factor: Коэффициент уровня шума (чем больше значение, тем сильнее шум)
    :return: Аудиофайл с добавленным шумом
    """
    logger.debug(f"Добавление шума с коэффициентом {noise_factor}.")
    noise = np.random.randn(len(data))
    return data + noise_factor * noise
