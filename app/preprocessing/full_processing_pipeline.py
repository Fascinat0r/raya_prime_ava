# app/preprocessing/full_processing_pipeline.py
# Description: Полный конвейер обработки данных для создания мел-спектрограмм и метаданных из аудиофайлов.

import concurrent.futures
import os
import time
from multiprocessing import Manager

import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from logger import get_logger
from preprocessing.augment import run_augmentation
from preprocessing.id_manager import SpectrogramIDManager
from preprocessing.process_wav_to_mel_spectrogram import process_audio_file

logger = get_logger("full_processing_pipeline")


def process_audio_to_spectrograms(raw_file_path: str, config: Config, overlap=0.0):
    """Обрабатывает аудио-файл до извлечения мел-спектрограмм с использованием новой функции."""
    mel_spectrograms = process_audio_file(
        raw_file_path,
        segment_length_seconds=10,
        target_segment_shape=config.SPECTROGRAM_SIZE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        overlap=overlap
    )
    return mel_spectrograms


def save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file, id_manager):
    """Сохраняет мел-спектрограммы и записывает метаданные для каждого сегмента."""
    value = os.path.basename(filename)[0]
    if value not in ['0', '1']:
        logger.error(f"Неверное значение целевого признака в файле {filename}")
        return None

    file_metadata = []
    os.makedirs(spectrograms_folder, exist_ok=True)

    for start_time, mel_spectrogram in mel_spectrograms:
        spectrogram_id = id_manager.get_next_spectrogram_id()
        logger.debug(f"Сохранение спектрограммы {spectrogram_id} из файла {filename}")
        spectrogram_filename = f"spectrogram_{spectrogram_id}.pt"
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_filename)

        with open(spectrogram_path, 'wb') as f:
            torch.save(mel_spectrogram, f)

        segment_metadata = {
            "value": value,
            "original_filename": filename,
            "spectrogram_filename": spectrogram_filename,
            "spectrogram_path": os.path.abspath(spectrogram_path),
            "start_time": start_time,  # Время теперь правильно рассчитывается в process_audio_file
            "source_type": "o"
        }
        file_metadata.append(segment_metadata)

    metadata_df = pd.DataFrame(file_metadata)
    if os.path.exists(metadata_file):
        metadata_df.to_csv(metadata_file, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_file, index=False)

    return file_metadata


def process_file(filename: str, config: Config, id_manager):
    raw_folder = config.RAW_FOLDER
    spectrograms_folder = config.SPECTROGRAMS_PATH
    metadata_file = config.METADATA_PATH
    filepath = os.path.join(raw_folder, filename)
    value = os.path.basename(filename)[0]
    overlap = 0.7 if value == "1" else 0.0

    # Обработка аудиофайла и получение списка мел-спектрограмм с их стартовыми временами
    mel_spectrograms = process_audio_to_spectrograms(filepath, config, overlap=overlap)
    if mel_spectrograms is None:
        return None

    # Сохранение спектрограмм и метаданных
    return save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file, id_manager)


def check_and_augment_data(metadata_file, config):
    """Проверяет баланс данных и запускает аугментацию при необходимости."""
    if not os.path.exists(metadata_file):
        logger.warning("Метаданные отсутствуют, аугментация невозможна.")
        return

    metadata = pd.read_csv(metadata_file)
    value_counts = metadata['value'].value_counts()

    if value_counts.min() / value_counts.max() < config.AUGMENTATION_THRESHOLD:
        logger.info("Запуск аугментации для балансировки классов.")
        run_augmentation(metadata, config)
    else:
        logger.info("Данные сбалансированы, аугментация не требуется.")


def run_parallel_processing(config):
    spectrograms_folder = config.SPECTROGRAMS_PATH
    metadata_file = config.METADATA_PATH

    if not os.path.exists(spectrograms_folder):
        os.makedirs(spectrograms_folder)

    with Manager() as manager:
        id_value = manager.Value('i', 1)  # Начальный ID (integer value, 'i' - для целых чисел)
        id_lock = manager.Lock()

        id_manager = SpectrogramIDManager(id_value, id_lock)
        id_manager.initialize_from_metadata(metadata_file)

        raw_files = [f for f in os.listdir(config.RAW_FOLDER) if f.endswith(".wav")]

        all_metadata = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.MAX_PREPROCESSING_PROCESSES) as executor:
            futures = {executor.submit(process_file, file, config, id_manager): file for file in raw_files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
                file_metadata = future.result()
                if file_metadata:
                    all_metadata.extend(file_metadata)

        logger.info(f"Метаданные сохранены в {metadata_file}")

    # Проверка и запуск аугментации при необходимости
    check_and_augment_data(metadata_file, config)


if __name__ == "__main__":
    config = Config()

    start_time = time.time()
    run_parallel_processing(config)
    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")
