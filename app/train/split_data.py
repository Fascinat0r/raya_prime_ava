# app/train/split_data.py
# Description: Функция для разделения мел-спектрограмм на тренировочную и тестовую выборки с равным соотношением значений value.
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_melspec_data(metadata_path, train_ratio=0.8, output_dir="./"):
    """
    Функция для разделения мел-спектрограмм на тренировочную и тестовую выборки с равным соотношением значений value.

    :param metadata_path: Путь к файлу метаданных CSV.
    :param train_ratio: Доля данных, используемая для тренировки (остальное - для теста).
    :param output_dir: Путь для сохранения разделенных данных (по умолчанию: "../data/split_data/").
    :return: None
    """
    # Загружаем метаданные
    df = pd.read_csv(metadata_path)

    # Проверка наличия выходной директории
    os.makedirs(output_dir, exist_ok=True)

    # Разделяем данные на основе значений 'value'
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for value in df['value'].unique():
        value_data = df[df['value'] == value]

        # Разделяем данные на тренировочные и тестовые с учетом соотношения
        value_train, value_test = train_test_split(value_data, train_size=train_ratio, random_state=42)

        # Добавляем в общий DataFrame
        train_data = pd.concat([train_data, value_train])
        test_data = pd.concat([test_data, value_test])

    # Сохраняем результаты в отдельные CSV файлы
    train_data.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)

    print(
        f"Данные успешно разделены! Тренировочные данные сохранены в {os.path.join(output_dir, 'train_metadata.csv')}, "
        f"тестовые данные в {os.path.join(output_dir, 'test_metadata.csv')}.")


# Пример использования:
if __name__ == "__main__":
    metadata_file_path = "../data/metadata.csv"  # Путь к файлу с метаданными
    split_melspec_data(metadata_file_path)
