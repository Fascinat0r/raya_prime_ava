import logging
import os

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from feature_extraction import extract_mfcc
from preprocessing import preprocess_audio


# Функция поиска совпадений в аудиофайлах
def find_matching_segments(reference_mfcc, folder_path, output_file, threshold=100):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Обработка файла: {file_path}")

            # Загружаем и обрабатываем аудиофайл
            try:
                y, sr = preprocess_audio(file_path)
                target_mfcc = extract_mfcc((y, sr))
            except Exception as e:
                logging.error(f"Ошибка загрузки или обработки файла {file_path}: {e}")
                continue

            # Сравнение эталонного и целевого MFCC
            distance, _ = fastdtw(reference_mfcc.T, target_mfcc.T, dist=euclidean)
            similarity = 100 * max(0, 1 - (distance / threshold))  # Процент совпадения

            if similarity > 50:  # Записываем только если совпадение выше 50%
                logging.info(f"Найдено совпадение в {filename} с уровнем {similarity:.2f}%")
                results[filename] = similarity
            else:
                logging.info(f"Совпадений не найдено в {filename}")

    # Запись результатов в файл
    with open(output_file, 'w') as f:
        for file, sim in results.items():
            f.write(f"Файл: {file}, Совпадение: {sim:.2f}%\n")
