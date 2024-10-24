import logging

from .triplet_loss import TripletLoss
from ..base_model import MelSpecNet

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MelSpecTripletLossNet(MelSpecNet):
    """
    Класс для сети, обучаемой с использованием триплетной потери (Triplet Loss) на базе мел-спектрограмм (MelSpec).
    """

    def __init__(self, margin):
        """
        Инициализация модели с триплетной потерей.
        :param margin: Отступ (маржин) для функции потерь Triplet Loss.
        """
        super().__init__()
        self.loss_layer = TripletLoss(margin)  # Используем слой для расчета триплетной потери
        logger.info(f"Инициализация модели с Triplet Loss, margin={margin}")

    def forward(self, anchor, positive, negative):
        """
        Прямой проход через сеть.
        :param anchor: Входные данные для якоря.
        :param positive: Входные данные для положительного примера.
        :param negative: Входные данные для отрицательного примера.
        :return: Выходные вектора для anchor, positive, и negative.
        """
        logger.debug("Выполнение прямого прохода через сеть.")

        n = anchor.shape[0]

        # Обработка якоря через сеть
        anchor_out = self.network(anchor)
        anchor_out = anchor_out.reshape(n, -1)
        anchor_out = self.linear_layer(anchor_out)
        logger.debug(f"Выходной вектор anchor: {anchor_out.shape}")

        # Обработка положительного примера через сеть
        positive_out = self.network(positive)
        positive_out = positive_out.reshape(n, -1)
        positive_out = self.linear_layer(positive_out)
        logger.debug(f"Выходной вектор positive: {positive_out.shape}")

        # Обработка отрицательного примера через сеть
        negative_out = self.network(negative)
        negative_out = negative_out.reshape(n, -1)
        negative_out = self.linear_layer(negative_out)
        logger.debug(f"Выходной вектор negative: {negative_out.shape}")

        return anchor_out, positive_out, negative_out

    def loss(self, anchor, positive, negative, reduction='mean'):
        """
        Вычисление триплетной потери.
        :param anchor: Векторные представления якоря.
        :param positive: Векторные представления положительного примера.
        :param negative: Векторные представления отрицательного примера.
        :param reduction: Метод агрегации потерь ('mean' или 'sum').
        :return: Значение потерь.
        """
        logger.debug(f"Вычисление потерь с reduction={reduction}")
        loss_val = self.loss_layer(anchor, positive, negative, reduction)
        logger.debug(f"Значение потерь: {loss_val.item()}")
        return loss_val
