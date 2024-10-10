import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from app.train.architecture import euclidean_distance, \
    cosine_similarity  # Импорт функции для корректной загрузки модели
from app.train.train import contrastive_loss
from app.utils.logger import get_logger

# Инициализация логгера
logger = get_logger("diagnostic")

# Пути к данным и модели
TRAIN_DATA_FILE = "../data/train/train_pairs.npz"
VALIDATION_DATA_FILE = "../data/train/validation_pairs.npz"
MODEL_PATH = "../data/models/siamese_model.keras"

# Размер входных данных для модели
INPUT_SHAPE = (157, 40, 1)  # (временные шаги, количество MFCC, 1 канал)

# Загрузка модели с пользовательской функцией
if os.path.exists(MODEL_PATH):
    logger.info(f"Загрузка модели из {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"euclidean_distance": euclidean_distance,
                                                                   "cosine_similarity": cosine_similarity,
                                                                   "contrastive_loss": contrastive_loss})
    logger.info("Модель успешно загружена!")


def load_data(file_path):
    """
    Загружает обучающие или валидационные данные из .npz файла.

    :param file_path: Путь к файлу данных.
    :return: Кортеж из трех элементов (X1, X2, y).
    """
    logger.info(f"Загрузка данных из {file_path}...")
    data = np.load(file_path)
    return data['X1'], data['X2'], data['y']


def check_labels_distribution(y_train, y_val):
    """
    Проверка распределения меток в тренировочных и валидационных данных.

    :param y_train: Метки тренировочных данных.
    :param y_val: Метки валидационных данных.
    """
    logger.info("Проверка распределения меток в данных...")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)

    logger.info(f"Распределение меток в тренировочных данных: {dict(zip(unique_train, counts_train))}")
    logger.info(f"Распределение меток в валидационных данных: {dict(zip(unique_val, counts_val))}")

    # Визуализация распределения
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.barplot(x=unique_train, y=counts_train)
    plt.title("Распределение меток в тренировочных данных")
    plt.subplot(1, 2, 2)
    sns.barplot(x=unique_val, y=counts_val)
    plt.title("Распределение меток в валидационных данных")
    plt.show()


def visualize_mfcc_pair(X1, X2, y, index=0):
    """
    Визуализация пары MFCC и их метки.

    :param X1: Первый элемент пары.
    :param X2: Второй элемент пары.
    :param y: Метка пары.
    :param index: Индекс пары для визуализации.
    """
    logger.info(f"Визуализация пары MFCC с индексом {index}. Метка: {y[index]}")

    mfcc1 = X1[index].reshape(X1.shape[1], X1.shape[2])
    mfcc2 = X2[index].reshape(X2.shape[1], X2.shape[2])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mfcc1.T, cmap='viridis', aspect='auto')
    plt.title("MFCC 1")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mfcc2.T, cmap='viridis', aspect='auto')
    plt.title(f"MFCC 2 (Метка: {y[index]})")
    plt.colorbar()
    plt.show()


def visualize_model_activations(model, X1, X2, index=0):
    """
    Визуализация активаций модели для одной пары.

    :param model: Обученная модель сиамской сети.
    :param X1: Первый элемент пары.
    :param X2: Второй элемент пары.
    :param index: Индекс пары для визуализации.
    """
    logger.info(f"Визуализация активаций модели для пары с индексом {index}...")

    # Извлечение активаций из модели
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
    activations = intermediate_layer_model.predict([X1[index:index + 1], X2[index:index + 1]])

    # Визуализация активаций
    for i, activation in enumerate(activations):
        if len(activation.shape) == 4:  # Если это сверточный слой
            plt.figure(figsize=(16, 4))
            plt.suptitle(f"Активации сверточного слоя {i + 1}")
            for j in range(min(activation.shape[-1], 8)):  # Показываем первые 8 карт признаков
                plt.subplot(1, 8, j + 1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')
                plt.axis('off')
            plt.show()


if __name__ == "__main__":
    # Загрузка тренировочных и валидационных данных
    X1_train, X2_train, y_train = load_data(TRAIN_DATA_FILE)
    X1_val, X2_val, y_val = load_data(VALIDATION_DATA_FILE)

    # Проверка распределения меток
    check_labels_distribution(y_train, y_val)

    # Визуализация случайных пар MFCC
    visualize_mfcc_pair(X1_train, X2_train, y_train, index=random.randint(0, len(y_train) - 1))
    visualize_mfcc_pair(X1_val, X2_val, y_val, index=random.randint(0, len(y_val) - 1))

    # Загрузка модели и визуализация активаций (если модель уже обучена)
    if os.path.exists(MODEL_PATH):
        logger.info(f"Загрузка модели из {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        visualize_model_activations(model, X1_train, X2_train, index=0)
