import os
import random

import pandas as pd
import torch

from logger import get_logger

logger = get_logger("augment")


def add_noise_to_spectrogram(mel_spectrogram, noise_factor=0.005):
    """
    Добавляет случайный шум к мел-спектрограмме для аугментации данных.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    noise_factor (float): Фактор шума (настраиваемый параметр). Чем больше, тем сильнее шум.

    Возвращает:
    Tensor: Аугментированная мел-спектрограмма с добавленным шумом.
    """
    noise = torch.randn(mel_spectrogram.size()) * noise_factor
    augmented_spectrogram = mel_spectrogram + noise
    return augmented_spectrogram


def run_augmentation(metadata, config):
    # Вычисление количества выборок для каждого класса
    class_counts = metadata['value'].value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()

    # Определение количества аугментированных выборок для балансировки
    target_augmentation_count = int((max_count - min_count) * config.AUGMENTATION_RATIO)
    minority_samples = metadata[metadata['value'] == class_counts.idxmin()]

    if target_augmentation_count > len(minority_samples):
        logger.warning(f"Целевое количество аугментации ({target_augmentation_count}) превышает доступное "
                       f"количество образцов ({len(minority_samples)}). Используем максимум доступных образцов.")
        target_augmentation_count = len(minority_samples)

    # Выбираем случайные спектрограммы для аугментации
    samples_to_augment = random.sample(list(minority_samples['spectrogram_path']), target_augmentation_count)

    # Сохраняем аугментированные спектрограммы и обновляем метаданные
    augmented_metadata = []
    for path in samples_to_augment:
        mel_spectrogram = torch.load(path)
        augmented_spectrogram = add_noise_to_spectrogram(mel_spectrogram, noise_factor=config.NOISE_FACTOR)

        # Создаём новое имя файла для аугментированной спектрограммы
        new_filename = path.replace("spectrogram_", "aug_spectrogram_")
        torch.save(augmented_spectrogram, new_filename)

        # Получаем метаинформацию для аугментированной спектрограммы
        segment_metadata = {
            "value": minority_samples['value'].iloc[0],  # Берём значение класса из исходного файла
            "original_filename": minority_samples['original_filename'].iloc[0],  # Имя исходного файла
            "spectrogram_filename": os.path.basename(new_filename),
            "spectrogram_path": os.path.abspath(new_filename),
            "source_type": "a"
        }
        augmented_metadata.append(segment_metadata)
        logger.debug(f"Сохранена аугментированная спектрограмма: {new_filename}")

    # Добавляем аугментированные данные в метаданные
    augmented_df = pd.DataFrame(augmented_metadata)
    if os.path.exists(config.METADATA_PATH):
        augmented_df.to_csv(config.METADATA_PATH, mode='a', header=False, index=False)
    else:
        augmented_df.to_csv(config.METADATA_PATH, index=False)

    logger.info("Аугментация завершена и метаданные обновлены.")
