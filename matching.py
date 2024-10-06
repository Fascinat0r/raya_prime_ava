import logging
import os

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from feature_extraction import extract_mfcc
from preprocessing import preprocess_audio


# Функция поиска совпадений в аудиофайлах
def find_matching_segments(reference_mfcc, folder_path, output_file, threshold=50):
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
            try:
                distance, _ = fastdtw(reference_mfcc.T, target_mfcc.T, dist=euclidean)
                similarity = 100 * max(0, 1 - (distance / threshold))  # Процент совпадения
            except Exception as e:
                logging.error(f"Ошибка при сравнении MFCC для файла {file_path}: {e}")
                continue

            if similarity > 10:  # Записываем только если совпадение выше 10%
                logging.info(f"Найдено совпадение в {filename} с уровнем {similarity:.2f}%")
                results[filename] = similarity
            else:
                logging.info(f"Совпадений не найдено в {filename}")

    # Запись результатов в файл
    try:
        with open(output_file, 'w') as f:
            for file, sim in results.items():
                f.write(f"Файл: {file}, Совпадение: {sim:.2f}%\n")
        logging.info(f"Результаты сохранены в файл: {output_file}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов в файл {output_file}: {e}")
