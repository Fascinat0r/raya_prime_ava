import os
import random

import pandas as pd

from augmentation import augment_file

# Пути к файлам и папкам
METADATA_FILE = "data/metadata.csv"
SEGMENTS_FOLDER = "data/segments"
AUGMENTED_FOLDER = "data/augmented"


def balance_dataset(metadata_file=METADATA_FILE, segments_folder=SEGMENTS_FOLDER, augmented_folder=AUGMENTED_FOLDER):
    """
    Балансировка сегментов целевого и других голосов с помощью аугментации.
    :param metadata_file: Путь к файлу с метаданными (metadata.csv).
    :param segments_folder: Путь к папке с существующими сегментами.
    :param augmented_folder: Папка для сохранения аугментированных сегментов.
    """
    # Загрузка метаданных
    metadata = pd.read_csv(metadata_file)

    # Разделение на целевой и нецелевой голос по имени файла (начинается с "1" или "0")
    target_segments = metadata[metadata['original_filename'].str.startswith("1")]
    other_segments = metadata[metadata['original_filename'].str.startswith("0")]

    print(f"Сегментов целевого голоса: {len(target_segments)}")
    print(f"Сегментов других голосов: {len(other_segments)}")

    # Определение меньшей категории
    if len(target_segments) > len(other_segments):
        category_to_augment = other_segments
        augment_type = "other"
    else:
        category_to_augment = target_segments
        augment_type = "target"

    # Количество недостающих сегментов для балансировки
    num_to_augment = abs(len(target_segments) - len(other_segments))
    print(f"Количество недостающих сегментов для балансировки: {num_to_augment}")

    # Убедиться, что папка для аугментации существует
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    # Создание новых сегментов с помощью аугментации
    new_metadata = []
    for _ in range(num_to_augment):
        # Выбор случайного сегмента из недостающей категории
        selected_segment = random.choice(category_to_augment['segment_path'].values)

        # Путь к аугментированному файлу
        original_filename = os.path.basename(selected_segment)
        augmented_filename = f"augmented_{random.randint(1000, 9999)}_{original_filename}"
        augmented_path = os.path.join(augmented_folder, augmented_filename)

        # Применение аугментации к выбранному файлу
        augment_file(selected_segment, augmented_path)

        # Обновление метаданных
        segment_metadata = {
            "original_filename": f"augmented_{augment_type}_{original_filename}",
            "segment_filename": augmented_filename,
            "segment_length_sec":
                metadata.loc[metadata['segment_path'] == selected_segment, 'segment_length_sec'].values[0],
            "overlap": metadata.loc[metadata['segment_path'] == selected_segment, 'overlap'].values[0],
            "segment_path": augmented_path
        }
        new_metadata.append(segment_metadata)

    # Добавление новых записей в метаданные
    metadata = metadata.append(new_metadata, ignore_index=True)

    # Сохранение обновленных метаданных
    metadata.to_csv(metadata_file, index=False)
    print(f"Обновленные метаданные сохранены в {metadata_file}. Добавлено {num_to_augment} новых сегментов.")


# Пример использования:
balance_dataset()
