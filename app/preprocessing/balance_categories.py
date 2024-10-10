import os
import random

import pandas as pd

from augmentation import augment_file

# Пути к файлам и папкам
METADATA_FILE = "data/metadata.csv"
SEGMENTS_FOLDER = "data/segments"
AUGMENTED_FOLDER = "data/augmented"


def get_next_segment_number(metadata, prefix):
    """
    Возвращает следующий номер сегмента для указанного префикса имени файла.
    :param metadata: DataFrame с метаданными.
    :param prefix: Префикс имени файла (например, "target" или "other").
    :return: Следующий номер сегмента.
    """
    existing_filenames = metadata['segment_filename'].tolist()
    segment_numbers = [int(f.split('_')[-1].replace('.wav', '')) for f in existing_filenames if f.startswith(prefix)]
    if segment_numbers:
        return max(segment_numbers) + 1
    else:
        return 1


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

    # Определение недостающей категории
    if len(target_segments) > len(other_segments):
        category_to_augment = other_segments
        augment_type = "other"
        prefix = "other_augmented"
    else:
        category_to_augment = target_segments
        augment_type = "target"
        prefix = "target_augmented"

    # Количество недостающих сегментов для балансировки
    num_to_augment = abs(len(target_segments) - len(other_segments))
    print(f"Количество недостающих сегментов для балансировки: {num_to_augment}")

    # Убедиться, что папка для аугментации существует
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    # Создание новых сегментов с помощью аугментации
    new_metadata = []
    next_segment_number = get_next_segment_number(metadata, prefix)

    for _ in range(num_to_augment):
        # Выбор случайного сегмента из недостающей категории
        selected_segment = random.choice(category_to_augment['segment_path'].values)

        # Генерация нового имени для аугментированного файла с логичной нумерацией
        original_filename = os.path.basename(selected_segment)
        augmented_filename = f"{prefix}_{next_segment_number:04d}.wav"
        augmented_path = os.path.join(augmented_folder, augmented_filename)

        # Применение аугментации к выбранному файлу
        augment_file(selected_segment, augmented_path)
        next_segment_number += 1  # Увеличение номера сегмента для следующего файла

        # Обновление метаданных
        segment_metadata = {
            "original_filename": f"{prefix}_{original_filename}",
            "segment_filename": augmented_filename,
            "segment_length_sec":
                metadata.loc[metadata['segment_path'] == selected_segment, 'segment_length_sec'].values[0],
            "overlap": metadata.loc[metadata['segment_path'] == selected_segment, 'overlap'].values[0],
            "segment_path": augmented_path
        }
        new_metadata.append(segment_metadata)

    # Преобразование нового списка метаданных в DataFrame
    new_metadata_df = pd.DataFrame(new_metadata)

    # Добавление новых записей в исходные метаданные с помощью pd.concat
    metadata = pd.concat([metadata, new_metadata_df], ignore_index=True)

    # Сохранение обновленных метаданных
    metadata.to_csv(metadata_file, index=False)
    print(f"Обновленные метаданные сохранены в {metadata_file}. Добавлено {num_to_augment} новых сегментов.")


# Пример использования:
balance_dataset()
