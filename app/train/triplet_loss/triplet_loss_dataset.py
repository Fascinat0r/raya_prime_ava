import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MelSpecTripletDataset(Dataset):
    """
    Класс для загрузки данных для обучения с использованием триплетной потери (Triplet Loss).
    Данные загружаются из файлов формата .pt на основе метаданных, хранящихся в CSV.
    """

    def __init__(self, metadata_file):
        """
        Инициализация датасета.
        :param metadata_file: Путь к файлу метаданных (CSV), содержащему информацию о спектрограммах и их классах.
        """
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
    def _pt_loader(path):
        """
        Вспомогательная функция для загрузки .pt файла и его преобразования в тензор PyTorch.
        """
        try:
            sample = torch.load(path)

            # Проверка формы данных
            assert sample.shape[0] == 1 and sample.shape[1] == 64 and sample.shape[
                2] == 64, "Неправильная форма данных!"
            logger.debug(f"Загрузка файла: {path}, Размер: {sample.shape}")
            return sample

        except Exception as e:
            logger.error(f"Ошибка при загрузке {path}: {e}")
            raise

    def __getitem__(self, index):
        """
        Возвращает триплет: якорь, положительный и отрицательный примеры.
        :param index: Индекс якорного примера.
        :return: (anchor, positive, negative).
        """
        # Получаем якорную спектрограмму
        row = self.metadata.iloc[index]
        anchor_path = row['spectrogram_path']
        anchor_label = row['value']
        anchor_x = self._pt_loader(anchor_path)

        # Положительный пример
        start, end = self.label_to_index_range[anchor_label]
        i = np.random.randint(low=start, high=end)
        positive_row = self.metadata.iloc[i]
        positive_x = self._pt_loader(positive_row['spectrogram_path'])

        # Отрицательный пример
        other_classes = list(range(self.num_classes))
        other_classes.remove(anchor_label)
        negative_label = np.random.choice(other_classes)
        start, end = self.label_to_index_range[negative_label]
        i = np.random.randint(low=start, high=end)
        negative_row = self.metadata.iloc[i]
        negative_x = self._pt_loader(negative_row['spectrogram_path'])

        return (anchor_x, anchor_label), (positive_x, anchor_label), (negative_x, negative_label)

    def __len__(self):
        """
        Возвращаем общую длину датасета.
        """
        return self.len_
