import librosa
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

from utils.my_logger import logger
from voice_model_generation.model import cosine_distance


def process_audio_segment(segment, model, reference_vector, sr=16000, n_mfcc=13):
    """
    Преобразует сегмент аудио в MFCC и определяет его сходство с эталонным вектором.
    Args:
        segment (ndarray): Аудиосегмент.
        model (tf.keras.Model): Загруженная модель нейросети.
        reference_vector (ndarray): Эталонный вектор голоса для сравнения.
        sr (int): Частота дискретизации.
        n_mfcc (int): Количество MFCC-признаков.

    Returns:
        float: Значение косинусного расстояния между векторами.
    """
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) > 50:
        mfcc = mfcc[:50]  # Приводим к фиксированному размеру
    elif len(mfcc) < 50:
        # Дополняем нулями до нужного размера
        mfcc = np.pad(mfcc, ((0, 50 - len(mfcc)), (0, 0)), mode='constant')

    mfcc = np.expand_dims(mfcc, axis=0)  # Добавляем batch dimension
    predicted_vector = model.predict(mfcc)

    return cosine(predicted_vector[0], reference_vector[0])


def get_audio_segments(audio, sr=16000, segment_length=2):
    """
    Разделяет аудиофайл на сегменты заданной длины с перекрытием.
    Args:
        audio (ndarray): Аудиосигнал.
        sr (int): Частота дискретизации.
        segment_length (int): Длина сегмента в секундах.

    Returns:
        list: Список аудиосегментов.
    """
    step = int(segment_length * sr)  # Длина сегмента в сэмплах
    overlap = step // 2  # Перекрытие 50%
    segments = [audio[i:i + step] for i in range(0, len(audio) - step, overlap)]
    return segments


def find_matching_segments(audio_path, model_path, reference_audio_path, threshold=0.3, segment_length=2):
    """
    Ищет все совпадения с целевым голосом в заданном аудиофайле.
    Args:
        audio_path (str): Путь к аудиофайлу для анализа.
        model_path (str): Путь к сохраненной модели.
        reference_audio_path (str): Путь к эталонному аудио с целевым голосом.
        threshold (float): Порог для косинусного расстояния.
        segment_length (int): Длина сегмента в секундах.

    Returns:
        list: Таймкоды совпадений в формате [(start_time, end_time, similarity)].
    """
    logger.info(f"Загрузка модели из {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects={"cosine_distance": cosine_distance})
    logger.info("Модель успешно загружена.")

    logger.info(f"Загрузка эталонного аудиофайла из {reference_audio_path}...")
    reference_audio, sr = librosa.load(reference_audio_path, sr=None)
    reference_mfcc = librosa.feature.mfcc(y=reference_audio, sr=sr, n_mfcc=13).T[:50]
    reference_vector = model.predict(np.expand_dims(reference_mfcc, axis=0))  # Эталонный вектор голоса

    logger.info(f"Загрузка аудиофайла для анализа из {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=16000)

    logger.info(f"Разделение аудиофайла на сегменты длиной {segment_length} секунд с перекрытием 50%...")
    segments = get_audio_segments(audio, sr=16000, segment_length=segment_length)
    logger.info(f"Всего сегментов для анализа: {len(segments)}")

    # Параллельная обработка сегментов для поиска совпадений
    logger.info("Начало параллельного анализа аудиосегментов...")
    similarities = Parallel(n_jobs=-1)(
        delayed(process_audio_segment)(segment, model, reference_vector) for segment in segments
    )

    # Преобразование результатов в таймкоды
    timecodes = []
    segment_step = segment_length / 2  # Шаг сегмента в секундах
    for idx, similarity in enumerate(similarities):
        if similarity < threshold:
            start_time = idx * segment_step
            end_time = start_time + segment_length
            timecodes.append((start_time, end_time, similarity))

    logger.info(f"Найдено совпадений: {len(timecodes)}")
    for start, end, sim in timecodes:
        logger.info(f"Совпадение: {start:.2f}s - {end:.2f}s, Сходство: {sim:.3f}")

    return timecodes


if __name__ == "__main__":
    # Пример использования:
    result = find_matching_segments(
        audio_path="audio_test.wav",
        model_path="models/voice_recognition_model.keras",
        reference_audio_path="reference_raya.wav",
        threshold=0.3,
        segment_length=2
    )
    print(f"Таймкоды совпадений: {result}")
