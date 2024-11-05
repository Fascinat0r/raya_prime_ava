# app/preprocessing/id_manager.py
# Description: Менеджер ID для уникальных идентификаторов мел-спектрограмм.

import os

import pandas as pd


class SpectrogramIDManager:
    def __init__(self, initial_id, lock):
        self.current_id = initial_id  # Shared memory object (Manager().Value)
        self.lock = lock  # Shared lock for synchronization

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
