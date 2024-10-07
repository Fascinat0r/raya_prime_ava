import random

import librosa
import numpy as np

from utils.my_logger import logger


def augment_audio(segments, target_length):
    """
    Увеличение количества сегментов путем дублирования и применения изменений.
    :param segments: Список исходных аудио-сегментов
    :param target_length: Необходимое количество сегментов
    :return: Список сегментов с увеличением количества
    """
    logger.info(
        f"Начало аугментации аудио-сегментов. Исходное количество: {len(segments)}, целевое количество: {target_length}")
    augmented_segments = segments.copy()
    while len(augmented_segments) < target_length:
        # Применяем аугментацию к случайному сегменту
        segment = random.choice(segments)

        # Разные типы аугментации
        augmentations = [
            change_pitch(segment, n_steps=random.choice([-2, -1, 1, 2])),  # Изменение тона
            change_speed(segment, rate=random.choice([0.9, 1.1, 1.2])),  # Изменение скорости
            add_noise(segment, noise_factor=0.005)  # Добавление шума
        ]

        # Выбираем случайную аугментацию и добавляем в список
        chosen_augmentation = random.choice(augmentations)
        augmented_segments.append(chosen_augmentation)
        logger.debug(
            f"Добавлен новый сегмент после аугментации. Текущее количество сегментов: {len(augmented_segments)}")

    logger.info(f"Завершена аугментация. Итоговое количество сегментов: {len(augmented_segments)}")
    return augmented_segments[:target_length]


def change_pitch(data, sr=16000, n_steps=2):
    """
    Изменение тона аудиофайла.
    :param data: Входной аудиофайл в виде массива numpy
    :param sr: Частота дискретизации
    :param n_steps: Количество шагов изменения тона (положительное или отрицательное значение)
    :return: Аудиофайл с измененным тоном
    """
    logger.debug(f"Изменение тона на {n_steps} шагов (частота дискретизации: {sr})")
    try:
        result = librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)
        logger.debug("Изменение тона успешно выполнено.")
        return result
    except Exception as e:
        logger.error(f"Ошибка изменения тона: {e}")
        return data


def change_speed(data, rate=1.0):
    """
    Изменение скорости воспроизведения аудиофайла.
    :param data: Входной аудиофайл в виде массива numpy
    :param rate: Коэффициент изменения скорости (меньше 1 — медленнее, больше 1 — быстрее)
    :return: Аудиофайл с измененной скоростью
    """
    logger.debug(f"Изменение скорости воспроизведения на коэффициент {rate}")
    try:
        result = librosa.effects.time_stretch(y=data, rate=rate)
        logger.debug("Изменение скорости успешно выполнено.")
        return result
    except Exception as e:
        logger.error(f"Ошибка изменения скорости: {e}")
        return data


def add_noise(data, noise_factor=0.005):
    """
    Добавление шума к аудиофайлу.
    :param data: Входной аудиофайл в виде массива numpy
    :param noise_factor: Коэффициент уровня шума (чем больше значение, тем сильнее шум)
    :return: Аудиофайл с добавленным шумом
    """
    logger.debug(f"Добавление шума к аудиосигналу с коэффициентом {noise_factor}")
    try:
        noise = np.random.randn(len(data))
        result = data + noise_factor * noise
        logger.debug("Шум успешно добавлен к аудиофайлу.")
        return result
    except Exception as e:
        logger.error(f"Ошибка добавления шума: {e}")
        return data
