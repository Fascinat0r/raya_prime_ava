import concurrent.futures
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from logger import get_logger
from preprocessing.convert import load_wav_file, tensor_to_audiosegment, audiosegment_to_numpy
from preprocessing.denoise_audio import denoise_audio_data
from preprocessing.filter_silent import remove_silence_from_data
from preprocessing.normalize_audio import normalize_audio_data
from preprocessing.process_wav_to_mel_spectrogram import process_audio_file
from root import PROJECT_ROOT

logger = get_logger("full_processing_pipeline")

# Ограничение ресурсов
MAX_PROCESSES = 4  # Максимальное количество одновременно выполняемых процессов
MAX_MEMORY_USAGE_MB = 4096  # Максимальное количество оперативной памяти в МБ

# Глобальный счетчик ID для мел-спектрограмм
global_spectrogram_id = 0


def initialize_global_spectrogram_id(metadata_file):
    """Инициализирует глобальный счетчик ID на основе существующих метаданных или с нуля."""
    global global_spectrogram_id

    if os.path.exists(metadata_file):
        try:
            metadata = pd.read_csv(metadata_file)
            existing_ids = metadata['spectrogram_filename'].str.extract(r'_(\d+)\.pt')[0].dropna().astype(int)
            if not existing_ids.empty:
                global_spectrogram_id = existing_ids.max() + 1
            else:
                global_spectrogram_id = 1
        except KeyError:
            global_spectrogram_id = 1
    else:
        global_spectrogram_id = 1


def get_next_spectrogram_id():
    """Возвращает следующий уникальный ID для мел-спектрограммы и увеличивает глобальный счетчик."""
    global global_spectrogram_id
    current_id = global_spectrogram_id
    global_spectrogram_id += 1
    return current_id


def process_audio_to_spectrograms(raw_file_path: str):
    """
    Обрабатывает аудио-файл до извлечения мел-спектрограмм без записи метаданных.
    :param raw_file_path: Путь к исходному аудио-файлу.
    :return: Список мел-спектрограмм.
    """

    # 1. Загрузка аудио и преобразование в AudioSegment
    audio_segment, sample_rate = load_wav_file(raw_file_path)
    # audio_segment = tensor_to_audiosegment(audio_segment, sample_rate)

    # 2. Удаление тишины
    # logger.info("Удаление тишины...")
    # nosilent_audio = remove_silence_from_data(audio_segment, silence_thresh=-40, min_silence_len=50)

    # 3. Нормализация
    # logger.info("Нормализация громкости...")
    # normalized_audio = normalize_audio_data(nosilent_audio, target_dBFS=-20.0)

    # 4. Преобразование в np.array для шумоподавления
    # logger.info("Шумоподавление...")
    # normalized_numpy = audiosegment_to_numpy(normalized_audio)
    # denoised_numpy = denoise_audio_data(normalized_numpy, normalized_audio.frame_rate, chunk_size=50000,
    #                                    prop_decrease=0.7, stationary=False, n_std_thresh_stationary=1.2)

    # 5. Преобразование в Tensor для мел-спектрограмм
    # denoised_tensor = torch.tensor(denoised_numpy).unsqueeze(0)  # Добавляем размерность канала

    # 6. Извлечение мел-спектрограмм с сегментацией
    mel_spectrograms = process_audio_file(raw_file_path, segment_length_seconds=10, target_segment_shape=(128, 128),
                                          n_fft=2048, hop_length=256, n_mels=128)

    return mel_spectrograms


def save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file):
    """
    Сохраняет мел-спектрограммы и записывает метаданные для каждого сегмента.
    :param mel_spectrograms: Список мел-спектрограмм.
    :param filename: Имя исходного файла.
    :param spectrograms_folder: Папка для сохранения мел-спектрограмм.
    :param metadata_file: Путь к файлу метаданных.
    :return: Метаданные для всех сегментов.
    """

    # Извлечение метки целевого признака из наименования файла, первый символ названия файла
    # 0 - негативный пример, 1 - позитивный пример
    value = os.path.basename(filename)[0]
    if value not in ['0', '1']:
        logger.error(f"Неверное значение целевого признака в файле {filename}")
        return None

    file_metadata = []
    os.makedirs(spectrograms_folder, exist_ok=True)

    for i, mel_spectrogram in enumerate(mel_spectrograms):
        spectrogram_id = get_next_spectrogram_id()
        spectrogram_filename = f"spectrogram_{spectrogram_id}.pt"
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_filename)

        # Сохранение мел-спектрограммы
        with open(spectrogram_path, 'wb') as f:
            torch.save(mel_spectrogram, f)

        # Запись метаданных
        segment_metadata = {
            "value": value,
            "original_filename": filename,
            "spectrogram_filename": spectrogram_filename,
            "spectrogram_path": os.path.abspath(spectrogram_path),
            "source_type": "o"
        }
        file_metadata.append(segment_metadata)

    # Запись метаданных в файл
    metadata_df = pd.DataFrame(file_metadata)
    if os.path.exists(metadata_file):
        metadata_df.to_csv(metadata_file, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_file, index=False)

    return file_metadata


def process_file(filename: str, raw_folder: str, spectrograms_folder: str, metadata_file: str):
    """Обрабатывает файл последовательно через все этапы и записывает метаданные для каждого сегмента."""
    filepath = os.path.join(raw_folder, filename)
    mel_spectrograms = process_audio_to_spectrograms(filepath)
    if mel_spectrograms is None:
        return None
    return save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file)


def run_parallel_processing(raw_folder, spectrograms_folder, metadata_file, max_processes=MAX_PROCESSES):
    """Запускает параллельную обработку всех файлов с аудио и извлекает мел-спектрограммы."""
    if not os.path.exists(spectrograms_folder):
        os.makedirs(spectrograms_folder)

    initialize_global_spectrogram_id(metadata_file)

    raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".wav")]

    all_metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(process_file, file, raw_folder, spectrograms_folder, metadata_file): file for file in
                   raw_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
            file_metadata = future.result()
            if file_metadata:
                all_metadata.extend(file_metadata)

    logger.info(f"Метаданные сохранены в {metadata_file}")


if __name__ == "__main__":
    raw_folder = os.path.join(PROJECT_ROOT, "data", "raw")  # Путь к исходной папке с аудио-файлами
    spectrograms_folder = os.path.join(PROJECT_ROOT, "data", "spectrograms")  # Путь для сохранения мел-спектрограмм
    metadata_file = os.path.join(PROJECT_ROOT, "data", "metadata.csv")  # Путь к файлу с метаданными

    start_time = time.time()
    run_parallel_processing(raw_folder, spectrograms_folder, metadata_file)
    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")
