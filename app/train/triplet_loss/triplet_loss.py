import logging

import torch
from torch import nn

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        """
        Инициализация модуля Triplet Loss.
        :param margin: Отступ для тройной потери.
        """
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()  # Используем косинусное сходство
        self.margin = margin  # Отступ для максимизации разности положительных и отрицательных расстояний

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings, reduction='mean'):
        """
        Прямой проход для вычисления тройной потери.
        :param anchor_embeddings: Вектора якоря (anchor).
        :param positive_embeddings: Положительные вектора (positive).
        :param negative_embeddings: Отрицательные вектора (negative).
        :param reduction: Метод агрегации потерь ('mean' или 'sum').
        :return: Значение потерь.
        """
        logger.debug("Вычисление тройной потери...")

        # Вычисляем расстояние как (1 - косинусное сходство)
        positive_distance = 1 - self.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - self.cosine_similarity(anchor_embeddings, negative_embeddings)

        # Лосс вычисляется как max(positive_distance - negative_distance + margin, 0)
        losses = torch.max(positive_distance - negative_distance + self.margin, torch.full_like(positive_distance, 0))

        if reduction == 'mean':
            loss_value = torch.mean(losses)
            logger.debug(f"Средняя потеря: {loss_value.item()}")
            return loss_value
        else:
            loss_value = torch.sum(losses)
            logger.debug(f"Суммарная потеря: {loss_value.item()}")
            return loss_value
