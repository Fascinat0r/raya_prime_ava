import os

import numpy as np
from sklearn.model_selection import train_test_split

from app.utils.logger import get_logger  # Подключение логгера

# Инициализация логгера
logger = get_logger("data_preparation")

# Параметры разбиения данных
TRAIN_SPLIT = 0.7  # Доля данных для обучения
VALIDATION_SPLIT = 0.2  # Доля данных для валидации (оставшиеся 30% делятся на 2)
TEST_SPLIT = 0.1  # Доля данных для тестирования
SEED = 42  # Случайное зерно для повторяемости результатов

# Пути к папкам
DATA_FOLDER = "../data/mfcc"  # Папка с MFCC-признаками
OUTPUT_FOLDER = "../data/datasets"  # Папка для сохранения выборок


def load_mfcc_data(folder):
    """
    Загружает MFCC данные из указанной папки.

    :param folder: Путь к папке с MFCC файлами.
    :return: Кортеж из двух элементов (X, y) — признаки и метки.
    """
    logger.info(f"Загрузка данных из папки: {folder}")
    X, y = [], []

    for file_name in os.listdir(folder):
        if file_name.endswith(".npy"):
            # Извлечение метки из имени файла (например, "1_example.npy" -> метка 1)
            label = int(file_name.split("_")[0])
            file_path = os.path.join(folder, file_name)

            # Загрузка MFCC признаков и добавление в массив
            mfcc = np.load(file_path)
            X.append(mfcc)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    logger.info(f"Загружено {X.shape[0]} примеров с размером {X.shape[1:]} и метками {set(y)}")
    return X, y


def split_data(X, y, train_size=TRAIN_SPLIT, validation_size=VALIDATION_SPLIT, seed=SEED):
    """
    Разделяет данные на обучающую, валидационную и тестовую выборки.

    :param X: Признаки (numpy массив).
    :param y: Метки (numpy массив).
    :param train_size: Доля данных для обучения.
    :param validation_size: Доля данных для валидации.
    :param seed: Случайное зерно для повторяемости.
    :return: Кортеж из шести элементов (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    logger.info("Разделение данных на обучающие, валидационные и тестовые выборки...")
    # Разделение на обучающую и оставшуюся выборку
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=seed, stratify=y)

    # Пропорциональное разделение оставшейся выборки на валидационную и тестовую
    validation_ratio = validation_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio, random_state=seed,
                                                    stratify=y_temp)

    logger.info(
        f"Размеры выборок:\n - Обучающая: {X_train.shape[0]} примеров\n - Валидационная: {X_val.shape[0]} примеров\n - Тестовая: {X_test.shape[0]} примеров")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, output_folder):
    """
    Сохраняет выборки в виде .npz файлов.

    :param X_train: Признаки обучающей выборки.
    :param X_val: Признаки валидационной выборки.
    :param X_test: Признаки тестовой выборки.
    :param y_train: Метки обучающей выборки.
    :param y_val: Метки валидационной выборки.
    :param y_test: Метки тестовой выборки.
    :param output_folder: Папка для сохранения выборок.
    """
    os.makedirs(output_folder, exist_ok=True)

    train_file = os.path.join(output_folder, "train_data.npz")
    val_file = os.path.join(output_folder, "validation_data.npz")
    test_file = os.path.join(output_folder, "test_data.npz")

    # Сохранение файлов
    np.savez(train_file, X=X_train, y=y_train)
    np.savez(val_file, X=X_val, y=y_val)
    np.savez(test_file, X=X_test, y=y_test)

    logger.info(
        f"Выборки успешно сохранены:\n - Обучающая: {train_file}\n - Валидационная: {val_file}\n - Тестовая: {test_file}")


if __name__ == "__main__":
    # Загрузка данных из папки с MFCC-признаками
    X, y = load_mfcc_data(DATA_FOLDER)

    # Разделение данных на обучающую, валидационную и тестовую выборки
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Сохранение выборок в .npz файлы
    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, OUTPUT_FOLDER)
    logger.info("Подготовка данных завершена.")
