# app/preprocessing/augment.py
# Description: Модуль для аугментации данных мел-спектрограмм.

import os
import random

import pandas as pd
import torch

from logger import get_logger

logger = get_logger("augment")


def add_white_noise(mel_spectrogram, noise_factor_range=(0.0, 0.1)):
    """
    Добавляет белый шум к мел-спектрограмме.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    noise_factor_range (tuple): Диапазон значений для фактора шума.

    Возвращает:
    Tensor: Аугментированная мел-спектрограмма.
    """
    noise_factor = random.uniform(*noise_factor_range)
    noise = torch.randn(mel_spectrogram.size()) * noise_factor
    return mel_spectrogram + noise


def add_pink_noise(mel_spectrogram, noise_factor_range=(0.0, 0.1)):
    """
    Добавляет розовый шум к мел-спектрограмме для аугментации.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    noise_factor_range (tuple): Диапазон значений для фактора шума.

    Возвращает:
    Tensor: Аугментированная мел-спектрограмма с розовым шумом.
    """
    noise_factor = random.uniform(*noise_factor_range)
    num_bins = mel_spectrogram.size(1)  # Количество частотных полос
    time_steps = mel_spectrogram.size(2)  # Количество временных фреймов

    # Создание розового шума по всем осям спектрограммы
    pink_noise = torch.randn(1, num_bins, time_steps) / torch.sqrt(
        torch.arange(1, num_bins + 1, dtype=torch.float).unsqueeze(1)
    )
    pink_noise = pink_noise.expand_as(mel_spectrogram) * noise_factor

    return mel_spectrogram + pink_noise


def change_volume(mel_spectrogram, volume_range=(0.8, 1.2)):
    """
    Изменяет громкость мел-спектрограммы на случайное значение.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    volume_range (tuple): Диапазон масштабирования громкости.

    Возвращает:
    Tensor: Мел-спектрограмма с изменённой громкостью.
    """
    volume_factor = random.uniform(*volume_range)
    return mel_spectrogram * volume_factor


def apply_random_augmentation(mel_spectrogram, config):
    """
    Применяет случайный метод аугментации к мел-спектрограмме.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    config (Config): Конфигурация с параметрами аугментации.

    Возвращает:
    Tensor: Аугментированная мел-спектрограмма.
    """
    augmentation_methods = [
        lambda x: add_white_noise(x, noise_factor_range=config.NOISE_FACTOR_RANGE),
        lambda x: add_pink_noise(x, noise_factor_range=config.NOISE_FACTOR_RANGE),
        lambda x: change_volume(x, volume_range=config.VOLUME_RANGE)
    ]
    augmentation_method = random.choice(augmentation_methods)
    return augmentation_method(mel_spectrogram)


def run_augmentation(metadata, config):
    """
    Выполняет аугментацию данных для балансировки классов.

    Аргументы:
    metadata (DataFrame): Исходные метаданные спектрограмм.
    config (Config): Конфигурация с параметрами аугментации и путями.
    """
    class_counts = metadata['value'].value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()

    target_augmentation_count = int((max_count - min_count) * config.AUGMENTATION_RATIO)
    minority_samples = metadata[metadata['value'] == class_counts.idxmin()]

    if target_augmentation_count > 0:
        logger.info(f"Необходимо добавить {target_augmentation_count} аугментированных образцов.")

    augmentations_needed = target_augmentation_count
    augmented_metadata = []

    while augmentations_needed > 0:
        batch_size = min(len(minority_samples), augmentations_needed)
        samples_to_augment = random.choices(list(minority_samples['spectrogram_path']), k=batch_size)

        for path in samples_to_augment:
            mel_spectrogram = torch.load(path)
            augmented_spectrogram = apply_random_augmentation(mel_spectrogram, config)

            new_filename = path.replace("spectrogram_", f"aug_spectrogram_{random.randint(1000, 9999)}_")
            torch.save(augmented_spectrogram, new_filename)

            original_metadata = minority_samples[minority_samples['spectrogram_path'] == path].iloc[0]
            segment_metadata = {
                "value": original_metadata['value'],
                "original_filename": original_metadata['original_filename'],
                "spectrogram_filename": os.path.basename(new_filename),
                "spectrogram_path": os.path.abspath(new_filename),
                "start_time": original_metadata['start_time'],
                "source_type": "a"
            }
            augmented_metadata.append(segment_metadata)
            logger.debug(f"Сохранена аугментированная спектрограмма: {new_filename}")

        augmentations_needed -= batch_size

    augmented_df = pd.DataFrame(augmented_metadata)
    if os.path.exists(config.METADATA_PATH):
        augmented_df.to_csv(config.METADATA_PATH, mode='a', header=False, index=False)
    else:
        augmented_df.to_csv(config.METADATA_PATH, index=False)

    logger.info("Аугментация завершена и метаданные обновлены.")
