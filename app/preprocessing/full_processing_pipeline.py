import concurrent.futures
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Manager, Lock, Value

from logger import get_logger
from preprocessing.convert import load_wav_file
from preprocessing.process_wav_to_mel_spectrogram import process_audio_file
from root import PROJECT_ROOT

logger = get_logger("full_processing_pipeline")

# Ограничение ресурсов
MAX_PROCESSES = 4  # Максимальное количество одновременно выполняемых процессов


class SpectrogramIDManager:
    def __init__(self, initial_id, lock):
        self.current_id = initial_id  # Это будет shared memory object (Manager().Value)
        self.lock = lock  # Это shared lock для синхронизации

    def initialize_from_metadata(self, metadata_file):
        """Инициализирует глобальный счетчик ID на основе существующих метаданных или с нуля."""
        if os.path.exists(metadata_file):
            try:
                metadata = pd.read_csv(metadata_file)
                existing_ids = metadata['spectrogram_filename'].str.extract(r'_(\d+)\.pt')[0].dropna().astype(int)
                if not existing_ids.empty:
                    with self.lock:  # Синхронизация доступа
                        self.current_id.value = existing_ids.max() + 1
            except KeyError:
                pass

    def get_next_spectrogram_id(self):
        """Возвращает следующий уникальный ID для мел-спектрограммы и увеличивает глобальный счетчик."""
        with self.lock:  # Синхронизация доступа для защиты общего ресурса
            current_id = self.current_id.value
            self.current_id.value += 1
        return current_id


def process_audio_to_spectrograms(raw_file_path: str):
    """Обрабатывает аудио-файл до извлечения мел-спектрограмм без записи метаданных."""
    audio_segment, sample_rate = load_wav_file(raw_file_path)
    mel_spectrograms = process_audio_file(raw_file_path, segment_length_seconds=10, target_segment_shape=(128, 128),
                                          n_fft=2048, hop_length=256, n_mels=128)
    return mel_spectrograms


def save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file, id_manager):
    """Сохраняет мел-спектрограммы и записывает метаданные для каждого сегмента."""
    value = os.path.basename(filename)[0]
    if value not in ['0', '1']:
        logger.error(f"Неверное значение целевого признака в файле {filename}")
        return None

    file_metadata = []
    os.makedirs(spectrograms_folder, exist_ok=True)

    for mel_spectrogram in mel_spectrograms:
        spectrogram_id = id_manager.get_next_spectrogram_id()
        logger.info(f"Сохранение спектрограммы {spectrogram_id} из файла {filename}")
        spectrogram_filename = f"spectrogram_{spectrogram_id}.pt"
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_filename)

        with open(spectrogram_path, 'wb') as f:
            torch.save(mel_spectrogram, f)

        segment_metadata = {
            "value": value,
            "original_filename": filename,
            "spectrogram_filename": spectrogram_filename,
            "spectrogram_path": os.path.abspath(spectrogram_path),
            "source_type": "o"
        }
        file_metadata.append(segment_metadata)

    metadata_df = pd.DataFrame(file_metadata)
    if os.path.exists(metadata_file):
        metadata_df.to_csv(metadata_file, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_file, index=False)

    return file_metadata


def process_file(filename: str, raw_folder: str, spectrograms_folder: str, metadata_file: str, id_manager):
    filepath = os.path.join(raw_folder, filename)
    mel_spectrograms = process_audio_to_spectrograms(filepath)
    if mel_spectrograms is None:
        return None
    return save_spectrogram_metadata(mel_spectrograms, filename, spectrograms_folder, metadata_file, id_manager)


def run_parallel_processing(raw_folder, spectrograms_folder, metadata_file, max_processes=MAX_PROCESSES):
    if not os.path.exists(spectrograms_folder):
        os.makedirs(spectrograms_folder)

    with Manager() as manager:
        # Создаем shared объекты: счетчик ID и Lock
        id_value = manager.Value('i', 1)  # Начальный ID (integer value, 'i' - для целых чисел)
        id_lock = manager.Lock()

        # Создаем объект менеджера ID
        id_manager = SpectrogramIDManager(id_value, id_lock)
        id_manager.initialize_from_metadata(metadata_file)

        raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".wav")]

        all_metadata = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = {
                executor.submit(process_file, file, raw_folder, spectrograms_folder, metadata_file, id_manager): file
                for file in
                raw_files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файлов"):
                file_metadata = future.result()
                if file_metadata:
                    all_metadata.extend(file_metadata)

        logger.info(f"Метаданные сохранены в {metadata_file}")


if __name__ == "__main__":
    raw_folder = os.path.join(PROJECT_ROOT, "data", "raw")
    spectrograms_folder = os.path.join(PROJECT_ROOT, "data", "spectrograms")
    metadata_file = os.path.join(PROJECT_ROOT, "data", "metadata.csv")

    start_time = time.time()
    run_parallel_processing(raw_folder, spectrograms_folder, metadata_file)
    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд.")
