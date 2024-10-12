import concurrent.futures
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from app.utils.logger import get_logger
from denoise_audio import denoise_audio_data
from filter_silent import remove_silence_from_data
from normalize_audio import normalize_audio_data
from process_wav_to_mel_spectrogram import process_audio_to_mel_spectrogram
from utils import audiosegment_to_numpy, load_wav_file, tensor_to_audiosegment

logger = get_logger("full_processing_pipeline")

# Папки для обработки
RAW_FOLDER = "../data/raw"
SPECTROGRAMS_FOLDER = "../data/spectrograms"
METADATA_FILE = "../data/metadata.csv"

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
            existing_ids = metadata['spectrogram_filename'].str.extract(r'_(\d+)\.npy')[0].dropna().astype(int)
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


def process_file(filename: str):
    """Обрабатывает файл последовательно через все этапы и записывает метаданные для каждого сегмента."""
    file_metadata = []

    # Путь к исходному файлу
    raw_file_path = os.path.join(RAW_FOLDER, filename)

    # 1. Загрузка аудио и преобразование в AudioSegment
    audio_segment, sample_rate = load_wav_file(raw_file_path)
    audio_segment = tensor_to_audiosegment(audio_segment, sample_rate)

    # 2. Удаление тишины
    logger.info("Удаление тишины...")
    nosilent_audio = remove_silence_from_data(audio_segment, silence_thresh=-40, min_silence_len=50)

    # 3. Нормализация
    logger.info("Нормализация громкости...")
    normalized_audio = normalize_audio_data(nosilent_audio, target_dBFS=-20.0)

    # 4. Преобразование в np.array для шумоподавления
    logger.info("Шумоподавление...")
    normalized_numpy = audiosegment_to_numpy(normalized_audio)
    denoised_numpy = denoise_audio_data(normalized_numpy, normalized_audio.frame_rate, chunk_size=50000,
                                        prop_decrease=0.7, stationary=False, n_std_thresh_stationary=1.2)

    # 5. Преобразование в Tensor для мел-спектрограмм
    denoised_tensor = torch.tensor(denoised_numpy).unsqueeze(0)  # Добавляем размерность канала

    # 6. Извлечение мел-спектрограмм с сегментацией
    mel_spectrograms = process_audio_to_mel_spectrogram(denoised_tensor, normalized_audio.frame_rate,
                                                        expected_shape=(64, 64), hop_length=512, n_fft=2048)

    # 7. Запись метаданных для каждой мел-спектрограммы
    for i, mel_spectrogram in enumerate(mel_spectrograms):
        spectrogram_id = get_next_spectrogram_id()
        spectrogram_filename = f"spectrogram_{spectrogram_id}.npy"
        spectrogram_path = os.path.join(SPECTROGRAMS_FOLDER, spectrogram_filename)

        # Сохранение мел-спектрограммы
        os.makedirs(SPECTROGRAMS_FOLDER, exist_ok=True)
        with open(spectrogram_path, 'wb') as f:
            torch.save(mel_spectrogram, f)

        # Запись метаданных
        segment_metadata = {
            "original_filename": filename,
            "spectrogram_filename": spectrogram_filename,
            "segment_length_sec": 5.0,
            "overlap": 0.5,
            "spectrogram_path": spectrogram_path,
            "source_type": "o"
        }
        file_metadata.append(segment_metadata)

    return file_metadata


def run_parallel_processing(max_processes=MAX_PROCESSES):
    """Запускает параллельную обработку всех файлов из папки data/raw с ограничением числа процессов."""
    if not os.path.exists(SPECTROGRAMS_FOLDER):
        os.makedirs(SPECTROGRAMS_FOLDER)

    initialize_global_spectrogram_id(METADATA_FILE)

    raw_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".wav")]

    all_metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(process_file, file): file for file in raw_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
            file_metadata = future.result()
            if file_metadata:
                all_metadata.extend(file_metadata)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)
    logger.info(f"Метаданные сохранены в {METADATA_FILE}")


if __name__ == "__main__":
    start_time = time.time()
    run_parallel_processing()
    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")
