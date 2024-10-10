import concurrent.futures
import os
import shutil
import time

import pandas as pd
from tqdm import tqdm

from augmentation import augment_file
from convert_to_mfcc import extract_mfcc
from denoise_audio import denoise_wav
from filter_silent import remove_silence
from normalize_audio import normalize_audio
from segment_audio import segment_audio

# Папки для обработки
RAW_FOLDER = "data/raw"
NOSILENT_FOLDER = "data/nosilent"
NORMALIZED_FOLDER = "data/normalized"
DENOISED_FOLDER = "data/denoised"
SEGMENTS_FOLDER = "data/segments"
AUGMENTED_FOLDER = "data/augmented"
MFCC_FOLDER = "data/mfcc"  # Общая папка для всех MFCC файлов
METADATA_FILE = "data/metadata.csv"

# Ограничение ресурсов
MAX_PROCESSES = 4  # Максимальное количество одновременно выполняемых процессов
MAX_MEMORY_USAGE_MB = 4096  # Максимальное количество оперативной памяти в МБ (например, 4 ГБ)

# Глобальный счетчик ID для MFCC файлов
global_mfcc_id = None


def initialize_global_mfcc_id(metadata_file):
    """Инициализирует глобальный счетчик ID на основе существующих метаданных или с нуля."""
    global global_mfcc_id

    # Если метаданные уже существуют, инициализировать счетчик на основе существующих данных
    if os.path.exists(metadata_file):
        try:
            metadata = pd.read_csv(metadata_file)
            existing_ids = metadata['mfcc_filename'].str.extract(r'_(\d+)\.npy')[0].dropna().astype(int)
            if not existing_ids.empty:
                global_mfcc_id = existing_ids.max() + 1
            else:
                global_mfcc_id = 1
        except KeyError:
            # Если столбца с именами MFCC еще нет, начать с 1
            global_mfcc_id = 1
    else:
        global_mfcc_id = 1


def clear_folders(folders):
    """Удаляет все файлы и папки внутри указанных директорий."""
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Удаляет всю папку и её содержимое
            os.makedirs(folder)  # Создает пустую папку заново
            print(f"Папка {folder} очищена.")


def get_next_mfcc_id():
    """Возвращает следующий уникальный ID для MFCC файла и увеличивает глобальный счетчик."""
    global global_mfcc_id
    current_id = global_mfcc_id
    global_mfcc_id += 1
    return current_id


def process_file(filename: str):
    """Обрабатывает файл последовательно через все этапы и записывает метаданные для каждого сегмента."""
    file_metadata = []

    # Путь к исходному файлу
    raw_file_path = os.path.join(RAW_FOLDER, filename)

    # 1. Удаление тишины
    nosilent_file_path = os.path.join(NOSILENT_FOLDER, filename)
    remove_silence(raw_file_path, nosilent_file_path, silence_thresh=-40, min_silence_len=500)

    # 2. Нормализация
    normalized_file_path = os.path.join(NORMALIZED_FOLDER, filename)
    normalize_audio(nosilent_file_path, normalized_file_path, target_dBFS=-20.0)

    # 3. Шумоподавление
    denoised_file_path = os.path.join(DENOISED_FOLDER, filename)
    denoise_wav(normalized_file_path, denoised_file_path, chunk_size=50000, prop_decrease=0.7, stationary=False,
                n_std_thresh_stationary=1.2)

    # 4. Сегментация
    segment_length = 5.0  # Длина сегмента в секундах
    overlap = 0.5  # Степень перекрытия
    segment_audio(denoised_file_path, SEGMENTS_FOLDER, segment_length=segment_length, overlap=overlap)

    # 5. Запись метаданных для каждого сегмента
    segment_files = [f for f in os.listdir(SEGMENTS_FOLDER) if f.startswith(os.path.splitext(filename)[0])]
    for segment_file in segment_files:
        segment_metadata = {
            "original_filename": filename,
            "segment_filename": segment_file,
            "segment_length_sec": segment_length,
            "overlap": overlap,
            "segment_path": os.path.join(SEGMENTS_FOLDER, segment_file),
            "source_type": "o"  # Обозначение для оригинальных сегментов
        }
        file_metadata.append(segment_metadata)

    return file_metadata


def augment_and_extract_mfcc(metadata_file=METADATA_FILE):
    """Выполняет аугментацию и преобразование сегментов в MFCC признаки."""
    metadata = pd.read_csv(metadata_file)

    augmented_metadata = []

    # 1. Аугментация сегментов
    for _, row in metadata.iterrows():
        segment_path = row['segment_path']
        augmented_filename = f"aug_{os.path.basename(segment_path)}"
        augmented_path = os.path.join(AUGMENTED_FOLDER, augmented_filename)

        # Применение аугментации
        augment_file(segment_path, augmented_path)

        # Добавление информации о новом сегменте
        augmented_metadata.append({
            "original_filename": row['original_filename'],
            "segment_filename": augmented_filename,
            "segment_length_sec": row['segment_length_sec'],
            "overlap": row['overlap'],
            "segment_path": augmented_path,
            "source_type": "a"  # Обозначение для аугментированных сегментов
        })

    # Обновление метаданных с аугментированными файлами
    augmented_df = pd.DataFrame(augmented_metadata)
    combined_metadata = pd.concat([metadata, augmented_df], ignore_index=True)

    # 2. Преобразование в MFCC сегментов и аугментированных файлов
    mfcc_metadata = []

    for _, row in combined_metadata.iterrows():
        segment_path = row['segment_path']

        # Определяем целевой ли это голос по оригинальному имени файла (начинается с "1" или "0")
        target_label = "1" if row['original_filename'].startswith("1") else "0"

        # Определяем тип сегмента (оригинальный или аугментированный)
        segment_type = row['source_type']  # "o" — original, "a" — augmented

        # Получаем следующий уникальный ID для MFCC файла
        mfcc_id = get_next_mfcc_id()
        mfcc_filename = f"{target_label}_{segment_type}_{mfcc_id}.npy"
        mfcc_output_path = os.path.join(MFCC_FOLDER, mfcc_filename)

        # Убедиться, что папка существует
        os.makedirs(MFCC_FOLDER, exist_ok=True)

        # Извлечение MFCC
        extract_mfcc(segment_path, mfcc_output_path)

        # Обновление метаданных
        row['mfcc_path'] = mfcc_output_path
        row['mfcc_filename'] = mfcc_filename
        mfcc_metadata.append(row)

    # Сохранение обновленных метаданных
    mfcc_metadata_df = pd.DataFrame(mfcc_metadata)
    mfcc_metadata_df.to_csv(metadata_file, index=False)
    print(f"Метаданные с MFCC признаками сохранены в {metadata_file}.")


def run_parallel_processing(max_processes=MAX_PROCESSES):
    """Запускает параллельную обработку всех файлов из папки data/raw с ограничением числа процессов."""
    for folder in [NOSILENT_FOLDER, NORMALIZED_FOLDER, DENOISED_FOLDER, SEGMENTS_FOLDER, AUGMENTED_FOLDER, MFCC_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    folders_to_clear = [
        NOSILENT_FOLDER, NORMALIZED_FOLDER, DENOISED_FOLDER, SEGMENTS_FOLDER, AUGMENTED_FOLDER, MFCC_FOLDER
    ]

    # Очистка всех целевых папок перед началом
    clear_folders(folders_to_clear)

    # Инициализация глобального ID для MFCC
    initialize_global_mfcc_id(METADATA_FILE)

    raw_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".wav")]

    all_metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(process_file, file): file for file in raw_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
            try:
                file_metadata = future.result()
                if file_metadata:
                    all_metadata.extend(file_metadata)
            except Exception as e:
                print(f"Ошибка при обработке файла {futures[future]}: {e}")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)
    print(f"Метаданные сохранены в {METADATA_FILE}")

    # Аугментация и преобразование в MFCC
    augment_and_extract_mfcc()


if __name__ == "__main__":
    start_time = time.time()
    run_parallel_processing()
    print(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")
