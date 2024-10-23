import argparse
import logging
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MelSpecCrossEntropyDataset(Dataset):
    """
    Класс для загрузки фильтробанок из файлов .npy для обучения модели с функцией потерь Cross Entropy.
    Использует файл метаданных для управления загрузкой данных.
    """

    def __init__(self, metadata_file):
        logger.info(f"Загрузка метаданных из файла: {metadata_file}")
        # Загружаем метаданные из CSV
        self.metadata = pd.read_csv(metadata_file)

        # Получаем количество классов из метаданных
        self.num_classes = self.metadata['value'].nunique()
        self.len_ = len(self.metadata)

        # Расчет количества образцов для каждого класса
        self.label_to_index_range = self._calculate_label_ranges()

        logger.info(f"Количество классов: {self.num_classes}, Количество образцов: {self.len_}")

    def _calculate_label_ranges(self):
        """
        Вспомогательная функция для расчета диапазонов индексов для каждого класса.
        """
        label_to_index_range = {}
        bin_counts = self.metadata['value'].value_counts().sort_index().values
        start = 0
        for i, count in enumerate(bin_counts):
            label_to_index_range[i] = (start, start + count)
            start += count
        return label_to_index_range

    @staticmethod
    def _npy_loader(path):
        """
        Вспомогательная функция для загрузки .npy файла и его преобразования в тензор PyTorch.
        """
        sample = np.load(path)
        assert sample.shape[0] == 64 and sample.shape[1] == 64 and sample.shape[2] == 1, "Неправильная форма данных!"
        logger.debug(f"Загрузка файла: {path}, Размер: {sample.shape}")

        # Преобразуем оси для соответствия ожиданиям PyTorch (канал на первой оси)
        sample = np.moveaxis(sample, 2, 0)
        sample = torch.from_numpy(sample).float()
        return sample

    def __getitem__(self, index):
        """
        Возвращаем данные и метку по индексу.
        """
        # Получаем путь к файлу и метку из метаданных
        row = self.metadata.iloc[index]
        spectrogram_path = row['spectrogram_path']
        label = row['value']

        # Загружаем спектрограмму
        data = self._npy_loader(spectrogram_path)
        return data, label

    def __len__(self):
        """
        Возвращаем общую длину датасета.
        """
        return self.len_

