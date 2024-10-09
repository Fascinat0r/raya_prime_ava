import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import multiprocessing
import random
import matplotlib.pyplot as plt
import librosa
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import load_model

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Загрузка и предобработка аудио
def load_and_preprocess_audio(filepath, target_sr=16000):
    """Загружает и преобразует частоту дискретизации к целевому значению."""
    audio, sr = librosa.load(filepath, sr=None)
    if sr != target_sr:
        logger.info(f"Преобразование частоты дискретизации {sr} -> {target_sr} для файла: {filepath}")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


# Разделение аудиофайла на сегменты
def split_audio_into_segments(audio, sr, segment_length=2, step=None):
    """Делит аудио на сегменты заданной длины."""
    if step is None:
        step = segment_length
    step_samples = int(step * sr)
    segment_samples = int(segment_length * sr)
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio) - segment_samples + 1, step_samples)]
    logger.info(f"Извлечено {len(segments)} сегментов с шагом {step} секунд.")
    return segments


# Извлечение MFCC для каждого сегмента
def extract_mfcc_features(segment, sr, n_mfcc=13):
    """Извлекает MFCC для одного сегмента."""
    if not isinstance(segment, np.ndarray):
        segment = np.array(segment)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).T
    return (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10)


# Сравнение сегментов
def compare_segments_with_target(siam_model, target_mfcc, test_mfcc):
    """Сравнивает каждый сегмент теста с целевыми сегментами."""
    results = []
    # Прогоняем каждый тестовый сегмент по каждому целевому
    for test in test_mfcc:
        distances = []
        for target in target_mfcc:
            target_input = np.expand_dims(target, axis=0)
            test_input = np.expand_dims(test, axis=0)
            # Получаем расстояние между одним целевым и тестовым сегментом
            distance = siam_model.predict([target_input, test_input])
            distances.append(distance)
        # Добавление значения среднего расстояния для каждого сегмента
        results.append(np.mean(distances) * 100)  # Преобразование в проценты
    return results


# Параллельное сравнение сегментов
def parallel_segment_comparison(siam_model, target_mfcc, test_segments, sr, n_mfcc, num_workers=4):
    """Параллельно извлекает MFCC и сравнивает сегменты."""
    with multiprocessing.Pool(num_workers) as pool:
        # Извлечение MFCC для всех тестовых сегментов в параллельном режиме
        test_mfcc = pool.starmap(extract_mfcc_features, [(seg, sr, n_mfcc) for seg in test_segments])
    # Сравниваем сегменты с помощью сиамской сети
    results = compare_segments_with_target(siam_model, target_mfcc, test_mfcc)
    return results


# Построение графика сходства с временной шкалой

import csv
from scipy.signal import savgol_filter


def plot_and_save_similarity_graph(results, segment_length, sr, title="Voice Similarity Over Time",
                                   output_file="similarity_results.csv"):
    """Строит график, где ось X — время (часы:минуты:секунды), ось Y — процент сходства и сохраняет результаты в CSV."""
    # Аппроксимация сигнала с использованием фильтра Савицкого-Голея
    smoothed_results = savgol_filter(results, window_length=11, polyorder=2)

    # Создание списка временных меток в секундах
    time_seconds = [i * segment_length for i in range(len(smoothed_results))]

    # Сохранение в CSV-файл
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (seconds)", "Similarity Percentage"])
        writer.writerows(zip(time_seconds, smoothed_results))

    # Построение графика
    time_labels = [str(timedelta(seconds=sec)) for sec in time_seconds]
    plt.figure(figsize=(15, 5))
    plt.plot(time_labels, smoothed_results, marker='o', linestyle='-', color='b', label='Smoothed Similarity')
    plt.xticks(rotation=45)
    plt.xlabel("Time (hh:mm:ss)")
    plt.ylabel("Similarity Percentage")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    logger.info(f"Результаты успешно сохранены в файл {output_file}")


# Демонстрационная функция предсказания и визуализации
def predict_and_visualize(siam_model, target_filepath, test_filepath, segment_length=2, n_segments=5, sr=16000,
                          threshold=0.5):
    """Проверяет целевой голос в тестовом аудио и визуализирует результаты."""
    # Загрузка и разделение целевого аудио
    target_audio, _ = load_and_preprocess_audio(target_filepath, sr)
    target_segments = split_audio_into_segments(target_audio, sr, segment_length)
    if len(target_segments) < n_segments:
        raise ValueError(f"Недостаточно сегментов в целевом аудио. Доступно: {len(target_segments)}")

    # Случайный выбор n сегментов
    target_segments = random.sample(target_segments, n_segments)
    target_mfcc = [extract_mfcc_features(seg, sr) for seg in target_segments]

    # Загрузка тестируемого аудио
    test_audio, _ = load_and_preprocess_audio(test_filepath, sr)
    test_segments = split_audio_into_segments(test_audio, sr, segment_length)

    # Параллельное сравнение
    results = parallel_segment_comparison(siam_model, target_mfcc, test_segments, sr, n_mfcc=13)

    logger.info(f"Результаты сравнения: {results}")

    # Визуализация результатов
    plot_and_save_similarity_graph(results, segment_length, sr, title="Voice Similarity Over Time")


# Запуск демонстрации
if __name__ == "__main__":
    # Импорт модели сиамской сети
    model_path = 'models/voice_recognition_model_v3.keras'
    model = load_model(model_path)

    path_to_target = "downloads/raya_references/Lp. Идеальный МИР #10 ЖИВОЙ РОБОТ • Майнкрафт.wav"
    path_to_test = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #10 ЖИВОЙ РОБОТ • Майнкрафт.wav"
    predict_and_visualize(model, path_to_target, path_to_test)
