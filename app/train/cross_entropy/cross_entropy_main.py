import logging
import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from app.train.cross_entropy.cross_entropy_dataset import MelSpecCrossEntropyDataset
from app.train.cross_entropy.cross_entropy_model import MelSpecCrossEntropyNet
from app.train.pt_utils import restore_model, restore_objects, save_model, save_objects
from config import Config

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_confusion_matrix(conf_matrix, classes):
    """
    Визуализирует матрицу ошибок.

    Аргументы:
    conf_matrix (ndarray): Матрица ошибок.
    classes (list): Список классов.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.show()


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    logger.info(f"Начало тренировки на эпохе {epoch}")

    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = model.loss(out, y)

        with torch.no_grad():
            pred = torch.argmax(out, dim=1)
            positive_accuracy += torch.sum(pred == y).item()
            negative_accuracy += torch.sum(pred != y).item()

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            logger.info(f"Эпоха {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
                        f"({100. * batch_idx / len(train_loader):.0f}%)]\tПотери: {loss.item():.6f}")

    positive_accuracy_mean = (100. * positive_accuracy) / len(train_loader.dataset)
    negative_accuracy_mean = (100. * negative_accuracy) / len(train_loader.dataset)
    logger.info(
        f"Точность на обучении: Positive: {positive_accuracy_mean:.2f}%, Negative: {negative_accuracy_mean:.2f}%")
    return np.mean(losses), positive_accuracy_mean, negative_accuracy_mean


def test(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0
    all_preds = []
    all_labels = []

    logger.info("Начало тестирования модели")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_loss_on = model.loss(out, y).item()
            losses.append(test_loss_on)

            pred = torch.argmax(out, dim=1)
            positive_accuracy += torch.sum(pred == y).item()
            negative_accuracy += torch.sum(pred != y).item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if log_interval is not None and batch_idx % log_interval == 0:
                logger.info(f"Тест: [{batch_idx * len(x)}/{len(test_loader.dataset)} "
                            f"({100. * batch_idx / len(test_loader):.0f}%)]\tПотери: {test_loss_on:.6f}")

    test_loss = np.mean(losses)
    positive_accuracy_mean = (100. * positive_accuracy) / len(test_loader.dataset)
    negative_accuracy_mean = (100. * negative_accuracy) / len(test_loader.dataset)

    logger.info(
        f"\nТестовый набор: Средние потери: {test_loss:.4f}, Positive Accuracy: {positive_accuracy}/{len(test_loader.dataset)} "
        f"({positive_accuracy_mean:.2f}%), Negative Accuracy: {negative_accuracy}/{len(test_loader.dataset)} ({negative_accuracy_mean:.2f}%)")

    # Расчет и вывод матрицы ошибок
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(conf_matrix, classes=['Negative', 'Positive'])

    return test_loss, positive_accuracy_mean, negative_accuracy_mean


def main(config: Config):
    model_path = config.MODEL_PATH
    use_cuda = config.USE_CUDA
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8, device=device)

    logger.info(f'Используемое устройство: {device}')

    import multiprocessing
    logger.info(f'Количество ядер процессора: {multiprocessing.cpu_count()}')

    kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}

    train_dataset = MelSpecCrossEntropyDataset('../train_metadata.csv', config.SPECTROGRAM_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, **kwargs)

    test_dataset = MelSpecCrossEntropyDataset('../test_metadata.csv', config.SPECTROGRAM_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, **kwargs)

    model = MelSpecCrossEntropyNet(reduction='mean').to(device)
    model = restore_model(model, model_path, device)

    # Восстанавливаем историю обучения с 8 элементами
    last_epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies, train_negative_accuracies, \
        test_positive_accuracies, test_negative_accuracies = restore_objects(
        model_path, (0, 0, [], [], [], [], [], []), device)

    start = last_epoch + 1 if max_accuracy > 0 else 0
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(start, start + config.EPOCHS):
        train_loss, train_positive_accuracy, train_negative_accuracy = train(
            model, device, train_loader, optimizer, epoch, log_interval=500)

        test_loss, test_positive_accuracy, test_negative_accuracy = test(
            model, device, test_loader)

        logger.info(f"Эпоха: {epoch}, Потери на обучении: {train_loss}, Потери на тестировании: {test_loss}, "
                    f"Точность на обучении: Positive: {train_positive_accuracy}, Negative: {train_negative_accuracy}, "
                    f"Точность на тестировании: Positive: {test_positive_accuracy}, Negative: {test_negative_accuracy}")

        # Сохраняем результаты
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_positive_accuracies.append(train_positive_accuracy)
        train_negative_accuracies.append(train_negative_accuracy)
        test_positive_accuracies.append(test_positive_accuracy)
        test_negative_accuracies.append(test_negative_accuracy)

        test_accuracy = (test_positive_accuracy + test_negative_accuracy) / 2

        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
        save_model(model, epoch, model_path)
        save_objects((epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies,
                      train_negative_accuracies, test_positive_accuracies, test_negative_accuracies),
                     epoch, model_path)
        logger.info(f"Сохранена модель эпохи {epoch} как контрольная точка.")


if __name__ == '__main__':
    config = Config()

    main(config)
