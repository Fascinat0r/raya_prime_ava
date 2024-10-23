import logging

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from app.train.cross_entropy.cross_entropy_dataset import MelSpecCrossEntropyDataset
from app.train.cross_entropy.cross_entropy_model import MelSpecCrossEntropyNet
from app.train.pt_utils import restore_model, restore_objects, save_model, save_objects

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    """
    Процесс тренировки модели за одну эпоху.

    :param model: Модель для обучения.
    :param device: Устройство (CPU или GPU), на котором происходит обучение.
    :param train_loader: Dataloader для тренировочных данных.
    :param optimizer: Оптимизатор (Adam).
    :param epoch: Текущая эпоха.
    :param log_interval: Интервал, через который выводятся лог-сообщения.
    :return: Среднее значение потерь и средняя точность за эпоху.
    """
    model.train()
    losses = []
    accuracy = 0

    logger.info(f"Начало тренировки на эпохе {epoch}")

    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Получаем предсказания от модели
        out = model(x)

        # Вычисляем потери с помощью loss-функции
        loss = model.loss(out, y)

        with torch.no_grad():
            pred = torch.argmax(out, dim=1)  # Получаем предсказания
            accuracy += torch.sum(pred == y)  # Считаем количество верных предсказаний

        losses.append(loss.item())
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Шаг оптимизатора

        if batch_idx % log_interval == 0:
            logger.info(f"Эпоха {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
                        f"({100. * batch_idx / len(train_loader):.0f}%)]\tПотери: {loss.item():.6f}")

    accuracy_mean = (100. * accuracy) / len(train_loader.dataset)
    logger.info(f"Точность на обучении: {accuracy_mean:.2f}%")
    return np.mean(losses), accuracy_mean


def test(model, device, test_loader, log_interval=None):
    """
    Тестирование модели на тестовых данных.

    :param model: Модель для тестирования.
    :param device: Устройство (CPU или GPU), на котором происходит тестирование.
    :param test_loader: Dataloader для тестовых данных.
    :param log_interval: Интервал логирования.
    :return: Среднее значение потерь и средняя точность на тестовых данных.
    """
    model.eval()
    losses = []
    accuracy = 0

    logger.info("Начало тестирования модели")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            x, y = x.to(device), y.to(device)

            # Получаем предсказания и вычисляем потери
            out = model(x)
            test_loss_on = model.loss(out, y).item()
            losses.append(test_loss_on)

            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum(pred == y)

            if log_interval is not None and batch_idx % log_interval == 0:
                logger.info(f"Тест: [{batch_idx * len(x)}/{len(test_loader.dataset)} "
                            f"({100. * batch_idx / len(test_loader):.0f}%)]\tПотери: {test_loss_on:.6f}")

    test_loss = np.mean(losses)
    accuracy_mean = (100. * accuracy) / len(test_loader.dataset)

    logger.info(f"\nТестовый набор: Средние потери: {test_loss:.4f}, Точность: {accuracy}/{len(test_loader.dataset)}"
                f" ({accuracy_mean:.2f}%)")
    return test_loss, accuracy_mean


def main():
    """
    Основная функция для запуска тренировки модели.
    Загружает данные, восстанавливает модель, выполняет обучение и сохраняет результаты.
    """
    model_path = 'saved_models_cross_entropy/'  # Путь для сохранения модели
    use_cuda = False  # Использование GPU, если доступен
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f'Используемое устройство: {device}')

    import multiprocessing
    logger.info(f'Количество ядер процессора: {multiprocessing.cpu_count()}')

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    # Загрузка тренировочного и тестового датасета с фильтробанками
    train_dataset = MelSpecCrossEntropyDataset('../train_metadata.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    test_dataset = MelSpecCrossEntropyDataset('../test_metadata.csv')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    # Загрузка модели
    model = MelSpecCrossEntropyNet(reduction='mean').to(device)
    model = restore_model(model, model_path)

    # Восстанавливаем историю обучения
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(
        model_path, (0, 0, [], [], [], []))

    start = last_epoch + 1 if max_accuracy > 0 else 0
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Цикл обучения по эпохам
    for epoch in range(start, start + 20):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, log_interval=500)
        test_loss, test_accuracy = test(model, device, test_loader)

        logger.info(f"Эпоха: {epoch}, Потери на обучении: {train_loss}, Потери на тестировании: {test_loss}, "
                    f"Точность на обучении: {train_accuracy}, Точность на тестировании: {test_accuracy}")

        # Сохраняем результаты
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Сохраняем модель, если достигнута лучшая точность
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies), epoch,
                         model_path)
            logger.info(f"Сохранена модель эпохи {epoch} как контрольная точка.")


if __name__ == '__main__':
    main()
