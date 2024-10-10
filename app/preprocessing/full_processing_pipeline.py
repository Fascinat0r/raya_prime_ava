import concurrent.futures
import os

import pandas as pd
import psutil
from tqdm import tqdm

# Импортирование ранее определенных функций
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
METADATA_FILE = "data/metadata.csv"

# Ограничение ресурсов
MAX_PROCESSES = 4  # Максимальное количество одновременно выполняемых процессов (уменьшить при перегрузке)
MAX_MEMORY_USAGE_MB = 4096  # Максимальное количество оперативной памяти в МБ (например, 4 ГБ)


def monitor_memory():
    """
    Проверяет, превышает ли текущее использование памяти заданное ограничение.
    :return: True, если использование памяти меньше предела, иначе False.
    """
    current_memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Использование в МБ
    return current_memory_usage < MAX_MEMORY_USAGE_MB


def process_file(filename: str):
    """
    Обрабатывает файл последовательно через все этапы: удаление тишины, нормализация, шумоподавление, сегментация.
    Сохраняет промежуточные и итоговые результаты в соответствующие папки и записывает метаданные.

    :param filename: Имя файла для обработки.
    :return: Список словарей с метаданными для каждого сегмента.
    """
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
            "segment_path": os.path.join(SEGMENTS_FOLDER, segment_file)
        }
        file_metadata.append(segment_metadata)

    return file_metadata


def run_parallel_processing(max_processes=MAX_PROCESSES):
    """
    Запускает параллельную обработку всех файлов из папки data/raw с ограничением числа процессов.
    Сохраняет метаданные о сегментах в файл metadata.csv.

    :param max_processes: Максимальное количество параллельных процессов.
    """
    # Убедиться, что все папки существуют
    for folder in [NOSILENT_FOLDER, NORMALIZED_FOLDER, DENOISED_FOLDER, SEGMENTS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Получение списка всех файлов в папке raw
    raw_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".wav")]

    # Параллельная обработка файлов
    all_metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(process_file, file): file for file in raw_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
            try:
                # Ожидание завершения задачи и получение результата
                file_metadata = future.result()
                if file_metadata:
                    all_metadata.extend(file_metadata)
            except Exception as e:
                print(f"Ошибка при обработке файла {futures[future]}: {e}")

    # Сохранение метаданных в CSV файл
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)
    print(f"Метаданные сохранены в {METADATA_FILE}")


if __name__ == "__main__":
    run_parallel_processing()
