import os
import random

import numpy as np

from app.utils.logger import get_logger  # Подключение логгера

# Инициализация логгера
logger = get_logger("data_preparation")

# Пути к папкам и файлам
MFCC_FOLDER = "../data/mfcc"
METADATA_FILE = "../data/metadata.csv"
OUTPUT_FILE_TRAIN = "../data/train/train_pairs.npz"
OUTPUT_FILE_VALIDATION = "../data/train/validation_pairs.npz"
OUTPUT_FILE_TEST = "../data/train/test_pairs.npz"

# Параметры разбиения
TRAIN_SPLIT = 0.7  # Доля данных для обучения
VALIDATION_SPLIT = 0.2  # Доля данных для валидации
TEST_SPLIT = 0.1  # Доля данных для тестирования
SEED = 42  # Случайное зерно для повторяемости результатов


def load_mfcc_files(mfcc_folder):
    """
    Загружает все MFCC файлы из указанной папки.

    :param mfcc_folder: Путь к папке с MFCC файлами.
    :return: Словарь с ключами вида (класс, тип), а значениями — списки путей к файлам.
    """
    logger.info(f"Загрузка MFCC файлов из папки: {mfcc_folder}")
    mfcc_files = {}
    for filename in os.listdir(mfcc_folder):
        if filename.endswith(".npy"):
            parts = filename.split("_")
            class_label = int(parts[0])  # 0 или 1
            file_type = parts[1]  # 'o' или 'a'

            if (class_label, file_type) not in mfcc_files:
                mfcc_files[(class_label, file_type)] = []
            mfcc_files[(class_label, file_type)].append(os.path.join(mfcc_folder, filename))

    logger.info(f"Всего загружено {len(mfcc_files)} категорий MFCC файлов.")
    return mfcc_files


def create_pairs(mfcc_files, num_pairs=10000):
    """
    Создает положительные и отрицательные пары из MFCC файлов.

    :param mfcc_files: Словарь с MFCC файлами, организованными по классам и типам.
    :param num_pairs: Количество пар, которое необходимо создать.
    :return: Два массива (X1, X2) и массив меток y.
    """
    logger.info(f"Создание {num_pairs} пар (положительных и отрицательных)...")
    X1, X2, y = [], [], []

    # Разделение файлов по классам
    target_files = mfcc_files.get((1, 'o'), []) + mfcc_files.get((1, 'a'), [])
    non_target_files = mfcc_files.get((0, 'o'), []) + mfcc_files.get((0, 'a'), [])

    # Создание положительных пар
    for _ in range(num_pairs // 2):
        x1 = random.choice(target_files)
        x2 = random.choice(target_files)
        X1.append(np.load(x1))
        X2.append(np.load(x2))
        y.append(1)

    # Создание отрицательных пар
    for _ in range(num_pairs // 2):
        x1 = random.choice(target_files)
        x2 = random.choice(non_target_files)
        X1.append(np.load(x1))
        X2.append(np.load(x2))
        y.append(0)

    logger.info(f"Пары успешно созданы: {len(X1)} пар.")
    return np.array(X1), np.array(X2), np.array(y)


def split_and_save_data(X1, X2, y, train_split=TRAIN_SPLIT, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT):
    """
    Разделяет данные на обучающую, валидационную и тестовую выборки и сохраняет их.

    :param X1: Первый элемент пары.
    :param X2: Второй элемент пары.
    :param y: Метки пар.
    :param train_split: Доля данных для обучения.
    :param validation_split: Доля данных для валидации.
    :param test_split: Доля данных для тестирования.
    """
    logger.info("Разделение данных на обучающие, валидационные и тестовые...")
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    X1, X2, y = X1[indices], X2[indices], y[indices]

    train_size = int(len(y) * train_split)
    validation_size = int(len(y) * validation_split)

    # Разделение на тренировочные, валидационные и тестовые данные
    X1_train, X2_train, y_train = X1[:train_size], X2[:train_size], y[:train_size]
    X1_val, X2_val, y_val = X1[train_size:train_size + validation_size], X2[train_size:train_size + validation_size], y[
                                                                                                                      train_size:train_size + validation_size]
    X1_test, X2_test, y_test = X1[train_size + validation_size:], X2[train_size + validation_size:], y[
                                                                                                     train_size + validation_size:]

    np.savez(OUTPUT_FILE_TRAIN, X1=X1_train, X2=X2_train, y=y_train)
    np.savez(OUTPUT_FILE_VALIDATION, X1=X1_val, X2=X2_val, y=y_val)
    np.savez(OUTPUT_FILE_TEST, X1=X1_test, X2=X2_test, y=y_test)
    logger.info(
        f"Данные успешно сохранены:\n - Обучающие: {OUTPUT_FILE_TRAIN}\n - Валидационные: {OUTPUT_FILE_VALIDATION}\n - Тестовые: {OUTPUT_FILE_TEST}")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    logger.info("Запуск подготовки данных...")

    mfcc_files = load_mfcc_files(MFCC_FOLDER)
    X1, X2, y = create_pairs(mfcc_files, num_pairs=10000)
    split_and_save_data(X1, X2, y)
    logger.info("Подготовка данных завершена.")
