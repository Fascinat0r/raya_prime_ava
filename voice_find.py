import logging
import os
import pickle

import librosa

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1. Извлечение и сохранение MFCC эталонного фрагмента
def extract_and_save_mfcc(reference_file, save_path):
    logging.info(f"Извлечение MFCC из эталонного файла: {reference_file}")
    y, sr = librosa.load(reference_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Сохранение MFCC в файл
    with open(save_path, 'wb') as f:
        pickle.dump(mfcc, f)
    logging.info(f"MFCC сохранены в файл: {save_path}")
    return mfcc


# 2. Загрузка MFCC из сохраненного файла
def load_mfcc(mfcc_file):
    logging.info(f"Загрузка MFCC из файла: {mfcc_file}")
    with open(mfcc_file, 'rb') as f:
        mfcc = pickle.load(f)
    return mfcc


# 3. Сравнение сегментов и расчет расстояния
def calculate_similarity(mfcc_ref, mfcc_target):
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    distance, _ = fastdtw(mfcc_ref.T, mfcc_target.T, dist=euclidean)
    return distance


# 4. Поиск совпадений по таймкодам с сохранением в файл
def find_matching_segments(reference_mfcc_file, folder_path, output_file, threshold=100):
    # Загружаем эталонный MFCC
    mfcc_reference = load_mfcc(reference_mfcc_file)

    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Обработка файла: {file_path}")

            y, sr = librosa.load(file_path)
            duration = librosa.get_duration(y=y, sr=sr)

            # Разделение аудио на сегменты
            segment_duration = 2  # Длительность сегмента в секундах
            step_size = segment_duration * sr

            matching_segments = []

            # Анализ каждого сегмента
            for start_sample in range(0, len(y), step_size):
                end_sample = min(start_sample + step_size, len(y))
                segment = y[start_sample:end_sample]

                # Извлечение MFCC для сегмента
                mfcc_segment = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

                # Сравнение с эталоном
                distance = calculate_similarity(mfcc_reference, mfcc_segment)

                # Сохранение таймкода при нахождении совпадения
                if distance < threshold:
                    start_time = start_sample / sr
                    end_time = end_sample / sr
                    matching_segments.append((start_time, end_time))

            # Сохранение таймкодов для каждого файла
            if matching_segments:
                results[filename] = matching_segments
                logging.info(f"Найдены совпадения в {filename}: {matching_segments}")

    # Сохранение результатов в файл
    with open(output_file, 'w') as f:
        for audio_file, segments in results.items():
            f.write(f"Файл: {audio_file}\n")
            for start, end in segments:
                f.write(f"Таймкоды: {start:.2f} - {end:.2f}\n")
            f.write("\n")

    logging.info(f"Результаты сохранены в файл: {output_file}")
    return results


# 5. Пример использования
reference_voice = "downloads\\raya_reference.mp3"
audio_folder = "E:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт"
mfcc_save_path = "cache\\saved_reference_mfcc.pkl"
output_file = "matching_segments.txt"

# Извлечение и сохранение эталонного MFCC (если нужно сделать один раз)
if not os.path.exists(mfcc_save_path):
    extract_and_save_mfcc(reference_voice, mfcc_save_path)

# Поиск совпадений и сохранение результатов в файл
results = find_matching_segments(mfcc_save_path, audio_folder, output_file)
