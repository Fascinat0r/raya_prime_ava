import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from train.pt_utils import restore_model, restore_objects, save_model, save_objects
from train.triplet_loss.triplet_loss_dataset import MelSpecTripletDataset
from train.triplet_loss.triplet_loss_model import MelSpecTripletLossNet

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_cosine_distance(a, b):
    """
    Функция для расчета косинусной дистанции между двумя тензорами.
    :param a: Тензор A.
    :param b: Тензор B.
    :return: Косинусная дистанция между A и B.
    """
    return 1 - F.cosine_similarity(a, b)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    """
    Функция для обучения модели за одну эпоху.
    :param model: Обучаемая модель.
    :param device: Устройство (CPU или GPU).
    :param train_loader: DataLoader для обучающей выборки.
    :param optimizer: Оптимизатор.
    :param epoch: Текущая эпоха.
    :param log_interval: Интервал для логирования прогресса.
    :return: Среднее значение потерь, точность на положительных и отрицательных примерах.
    """
    model.train()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    positive_distances = []
    negative_distances = []

    for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(train_loader)):
        ax, px, nx = ax.to(device), px.to(device), nx.to(device)
        optimizer.zero_grad()

        # Прямой проход
        a_out, p_out, n_out = model(ax, px, nx)
        loss = model.loss(a_out, p_out, n_out)
        losses.append(loss.item())

        # Вычисляем расстояния и точность
        with torch.no_grad():
            p_distance = _get_cosine_distance(a_out, p_out)
            positive_distances.append(torch.mean(p_distance).item())

            n_distance = _get_cosine_distance(a_out, n_out)
            negative_distances.append(torch.mean(n_distance).item())

            positive_distance_mean = np.mean(positive_distances)
            negative_distance_mean = np.mean(negative_distances)

            positive_std = np.std(positive_distances)
            threshold = positive_distance_mean + 3 * positive_std

            positive_results = p_distance < threshold
            positive_accuracy += torch.sum(positive_results).item()

            negative_results = n_distance >= threshold
            negative_accuracy += torch.sum(negative_results).item()

        # Обратное распространение и шаг оптимизатора
        loss.backward()
        optimizer.step()

        # Логирование промежуточных результатов
        if batch_idx % log_interval == 0:
            logger.info('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(ax), len(train_loader.dataset),
                                                100. * batch_idx / len(train_loader), loss.item()))
            logger.info(
                'Train Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}'.format(
                    positive_distance_mean, negative_distance_mean, positive_std, threshold))

    positive_accuracy_mean = 100. * positive_accuracy / len(train_loader.dataset)
    negative_accuracy_mean = 100. * negative_accuracy / len(train_loader.dataset)
    return np.mean(losses), positive_accuracy_mean, negative_accuracy_mean


def test(model, device, test_loader, log_interval=None):
    """
    Функция для тестирования модели.
    :param model: Тестируемая модель.
    :param device: Устройство (CPU или GPU).
    :param test_loader: DataLoader для тестовой выборки.
    :param log_interval: Интервал для логирования прогресса.
    :return: Среднее значение потерь, точность на положительных и отрицательных примерах.
    """
    model.eval()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    positive_distances = []
    negative_distances = []

    with torch.no_grad():
        for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(test_loader)):
            ax, px, nx = ax.to(device), px.to(device), nx.to(device)

            # Прямой проход
            a_out, p_out, n_out = model(ax, px, nx)
            test_loss_on = model.loss(a_out, p_out, n_out, reduction='mean').item()
            losses.append(test_loss_on)

            # Вычисляем расстояния и точность
            p_distance = _get_cosine_distance(a_out, p_out)
            positive_distances.append(torch.mean(p_distance).item())

            n_distance = _get_cosine_distance(a_out, n_out)
            negative_distances.append(torch.mean(n_distance).item())

            positive_distance_mean = np.mean(positive_distances)
            negative_distance_mean = np.mean(negative_distances)

            positive_std = np.std(positive_distances)
            threshold = positive_distance_mean + 3 * positive_std

            positive_results = p_distance < threshold
            positive_accuracy += torch.sum(positive_results).item()

            negative_results = n_distance >= threshold
            negative_accuracy += torch.sum(negative_results).item()

            # Логирование
            if log_interval is not None and batch_idx % log_interval == 0:
                logger.info('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()), batch_idx * len(ax), len(test_loader.dataset),
                                             100. * batch_idx / len(test_loader), test_loss_on))

    test_loss = np.mean(losses)
    positive_accuracy_mean = 100. * positive_accuracy / len(test_loader.dataset)
    negative_accuracy_mean = 100. * negative_accuracy / len(test_loader.dataset)

    logger.info('Test Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}'.format(
        positive_distance_mean, negative_distance_mean, positive_std, threshold))

    logger.info(
        '\nTest set: Average loss: {:.4f}, Positive Accuracy: {}/{} ({:.0f}%), Negative Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, positive_accuracy, len(test_loader.dataset), positive_accuracy_mean,
            negative_accuracy, len(test_loader.dataset), negative_accuracy_mean))

    return test_loss, positive_accuracy_mean, negative_accuracy_mean


def main():
    """
    Основная функция для настройки и запуска обучения и тестирования модели.
    """
    model_path = 'siamese_melspec_saved/'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info(f'Используемое устройство: {device}')

    # Ограничиваем использование GPU памяти, если доступен CUDA
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8, device=device)

    # Логирование числа доступных ядер процессора
    import multiprocessing
    logger.info(f'Количество ядер процессора: {multiprocessing.cpu_count()}')

    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if use_cuda else {}

    # Загрузка обучающего и тестового датасетов
    train_dataset = MelSpecTripletDataset('../train_metadata.csv')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, **kwargs)

    test_dataset = MelSpecTripletDataset('../test_metadata.csv')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, **kwargs)

    # Загрузка модели и истории обучения
    model = MelSpecTripletLossNet(margin=0.2).to(device)
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies, train_negative_accuracies, \
        test_positive_accuracies, test_negative_accuracies = restore_objects(model_path, (0, 0, [], [], [], [], [], []))

    start = last_epoch + 1 if max_accuracy > 0 else 0
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Цикл обучения по эпохам
    for epoch in range(start, start + 20):
        train_loss, train_positive_accuracy, train_negative_accuracy = train(
            model, device, train_loader, optimizer, epoch, log_interval=500)

        test_loss, test_positive_accuracy, test_negative_accuracy = test(model, device, test_loader)

        logger.info(f'After epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}, '
                    f'train positive accuracy: {train_positive_accuracy}, train negative accuracy: {train_negative_accuracy}, '
                    f'test positive accuracy: {test_positive_accuracy}, test negative accuracy: {test_negative_accuracy}')

        # Сохранение результатов
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_positive_accuracies.append(train_positive_accuracy)
        test_positive_accuracies.append(test_positive_accuracy)
        train_negative_accuracies.append(train_negative_accuracy)
        test_negative_accuracies.append(test_negative_accuracy)

        test_accuracy = (test_positive_accuracy + test_negative_accuracy) / 2

        # Сохраняем модель, если достигнута лучшая точность
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies,
                          train_negative_accuracies, test_positive_accuracies, test_negative_accuracies),
                         epoch, model_path)
            logger.info(f'Сохранена модель эпохи {epoch} как контрольная точка.')


if __name__ == '__main__':
    main()