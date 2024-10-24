import logging

from torch import nn

from app.train.base_model import MelSpecNet

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MelSpecCrossEntropyNet(MelSpecNet):
    """
    Модель на основе MelSpecNet, которая использует функцию потерь Cross Entropy для классификации.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        # Используем слой с функцией потерь Cross Entropy
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        """
        Прямой проход через модель.
        :param x: Входные данные (тензор).
        :return: Прогнозы модели.
        """
        logger.debug(f"Размер входа: {x.shape}")

        # Пропускаем через сверточную сеть
        n = x.shape[0]  # Количество образцов в батче
        out = self.network(x)
        logger.debug(f"Размер после сверточной сети: {out.shape}")

        # Преобразуем выходные данные в линейный слой
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        logger.debug(f"Размер выхода модели: {out.shape}")
        return out

    def loss(self, predictions, labels):
        """
        Вычисление потерь модели.
        :param predictions: Прогнозы модели.
        :param labels: Истинные метки классов.
        :return: Значение потерь.
        """
        loss_val = self.loss_layer(predictions, labels)
        logger.debug(f"Потери: {loss_val.item()}")
        return loss_val
