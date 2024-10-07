import logging
import os

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from feature_extraction import extract_mfcc
from preprocessing import preprocess_audio


# Поиск совпадений с использованием MFCC и DTW
def find_matching_segments(reference_mfcc, folder_path, output_file, threshold=50):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Обработка файла: {file_path}")
            try:
                y, sr = preprocess_audio(file_path)
                target_mfcc = extract_mfcc((y, sr))
            except Exception as e:
                logging.error(f"Ошибка обработки {file_path}: {e}")
                continue

            # Сравнение MFCC
            distance, _ = fastdtw(reference_mfcc.T, target_mfcc.T, dist=euclidean)
            similarity = 100 * max(0, 1 - (distance / threshold))
            if similarity > 10:
                results[filename] = similarity

    with open(output_file, 'w') as f:
        for file, sim in results.items():
            f.write(f"{file}: {sim:.2f}%\n")
