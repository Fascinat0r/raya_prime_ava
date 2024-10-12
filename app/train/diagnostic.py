import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к папке с MFCC файлами
MFCC_FOLDER = "../data/mfcc"

def load_all_mfcc_files(mfcc_folder):
    """
    Загружает все MFCC файлы из указанной папки и возвращает список массивов.

    :param mfcc_folder: Путь к папке с MFCC файлами.
    :return: Список массивов MFCC.
    """
    mfcc_list = []
    file_shapes = []  # Список для хранения форм файлов
    for filename in os.listdir(mfcc_folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(mfcc_folder, filename)
            mfcc = np.load(file_path)
            mfcc_list.append(mfcc)
            file_shapes.append(mfcc.shape)
    return mfcc_list, file_shapes

def analyze_mfcc_shapes(file_shapes):
    """
    Анализ форм всех MFCC файлов.

    :param file_shapes: Список форм всех файлов.
    """
    # Считаем количество файлов каждой формы
    unique_shapes, counts = np.unique(file_shapes, return_counts=True)
    print(f"Уникальные формы MFCC: {dict(zip(unique_shapes, counts))}")

    # Проверка на наличие неоднородных форм
    if len(unique_shapes) > 1:
        print(f"Внимание! Найдено несколько уникальных форм: {unique_shapes}")
    else:
        print(f"Все MFCC файлы имеют одинаковую форму: {unique_shapes[0]}")

def visualize_mfcc_distribution(mfcc_list):
    """
    Визуализация статистических характеристик MFCC для всех файлов.

    :param mfcc_list: Список массивов MFCC.
    """
    means = [np.mean(mfcc) for mfcc in mfcc_list]
    stds = [np.std(mfcc) for mfcc in mfcc_list]

    plt.figure(figsize=(12, 5))

    # Гистограмма средних значений MFCC
    plt.subplot(1, 2, 1)
    plt.hist(means, bins=30, color='skyblue')
    plt.title("Распределение средних значений MFCC")
    plt.xlabel("Среднее значение")
    plt.ylabel("Количество файлов")

    # Гистограмма стандартных отклонений MFCC
    plt.subplot(1, 2, 2)
    plt.hist(stds, bins=30, color='salmon')
    plt.title("Распределение стандартных отклонений MFCC")
    plt.xlabel("Стандартное отклонение")
    plt.ylabel("Количество файлов")

    plt.show()

def plot_random_mfcc_heatmaps(mfcc_list, num_samples=5):
    """
    Визуализация тепловых карт случайных MFCC файлов.

    :param mfcc_list: Список массивов MFCC.
    :param num_samples: Количество случайных файлов для визуализации.
    """
    plt.figure(figsize=(15, num_samples * 3))
    for i in range(num_samples):
        mfcc = random.choice(mfcc_list)
        plt.subplot(num_samples, 1, i + 1)
        sns.heatmap(mfcc.T, cmap='viridis', cbar=True)
        plt.title(f"Тепловая карта MFCC (Пример {i + 1})")
    plt.show()

def check_for_missing_values(mfcc_list):
    """
    Проверка на наличие пропущенных значений (NaN или Inf) в данных MFCC.

    :param mfcc_list: Список массивов MFCC.
    """
    has_nan = any(np.isnan(mfcc).any() for mfcc in mfcc_list)
    has_inf = any(np.isinf(mfcc).any() for mfcc in mfcc_list)

    if has_nan:
        print("Предупреждение: В некоторых файлах MFCC присутствуют NaN значения!")
    else:
        print("NaN значения не найдены.")

    if has_inf:
        print("Предупреждение: В некоторых файлах MFCC присутствуют Inf значения!")
    else:
        print("Inf значения не найдены.")

if __name__ == "__main__":
    # Загрузка всех MFCC файлов
    mfcc_files, file_shapes = load_all_mfcc_files(MFCC_FOLDER)

    # Анализ форм файлов
    analyze_mfcc_shapes(file_shapes)

    # Визуализация распределения средних значений и стандартных отклонений
    visualize_mfcc_distribution(mfcc_files)

    # Визуализация тепловых карт для случайных MFCC
    plot_random_mfcc_heatmaps(mfcc_files, num_samples=5)

    # Проверка на наличие NaN и Inf значений
    check_for_missing_values(mfcc_files)
