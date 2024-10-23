import logging
from abc import abstractmethod

from torch import nn

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MelSpecResBlock(nn.Module):
    """
    Реснет-блок для обработки мел-спектрограмм.
    Блок состоит из двух сверточных слоев с ReLU активацией и остаточным соединением.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Рассчитываем padding для сохранения размерности после сверточных операций
        padding = (kernel_size - 1) // 2

        # Основная сеть: два сверточных слоя с пакетной нормализацией и активацией
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Проход данных через сеть и добавление остаточного соединения
        :param x: Входные данные.
        :return: Выход сети.
        """
        out = self.network(x)
        out = out + x  # Добавляем остаточное соединение
        out = self.relu(out)
        logger.debug(f"Размер выхода после ResBlock: {out.shape}")
        return out


class MelSpecNet(nn.Module):
    """
    Основная модель MelSpecNet, использующая несколько блоков ResNet для обработки фильтробанок.
    Сеть уменьшает размерность изображения и преобразует его в вектор эмбеддинга.
    """

    def __init__(self):
        """
        Инициализация сети MelSpecNet.
        """
        super().__init__()

        # Определение сети с использованием сверточных слоев и блоков ResNet
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            MelSpecResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            MelSpecResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            MelSpecResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            MelSpecResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)  # Пуллинг для уменьшения размерности перед линейным слоем
        )

        # Линейный слой для получения финального вектора эмбеддинга
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)  # Преобразуем вектор в эмбеддинг размером 250
        )

    @abstractmethod
    def forward(self, *input_):
        """
        Проход данных через сеть. Абстрактный метод.
        :param input_: Входные данные.
        :return: Выход сети.
        """
        raise NotImplementedError('Call one of the subclasses of this class')
