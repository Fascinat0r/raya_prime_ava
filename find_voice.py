import librosa
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from tensorflow.keras.models import load_model

from utils.my_logger import logger


# Функция для вычисления MFCC признаков
def extract_mfcc(audio_segment, sr=16000, n_mfcc=13):
    """Вычисляет MFCC признаки для аудиосегмента."""
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc).T[:50]
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
    return mfcc_norm


# Разделение аудиофайла на сегменты
def split_audio(audio, sr, segment_length=2):
    """Разделяет аудиофайл на сегменты фиксированной длины."""
    step = int(segment_length * sr)
    segments = [audio[i:i + step] for i in range(0, len(audio) - step, step // 2)]
    return segments


# Функция для предсказания совпадения голосов
def predict_with_model(model, segment_mfcc, reference_mfcc):
    """Применяет модель для предсказания совпадения сегмента с эталонным голосом."""
    # Создаём батч для пары (эталонный голос, тестовый сегмент)
    reference_batch = np.repeat(np.expand_dims(reference_mfcc, axis=0), len(segment_mfcc), axis=0)
    predictions = model.predict([reference_batch, segment_mfcc])
    return predictions


# Основная функция для обработки аудиофайла и демонстрации работы модели
def demo_voice_recognition(model_path, reference_file, test_file, segment_length=2, sr=16000):
    """Демонстрирует работу модели на тестовом аудиофайле."""
    logger.info(f"Загрузка эталонного аудиофайла: {reference_file}")
    ref_audio, _ = librosa.load(reference_file, sr=sr)
    ref_segments = split_audio(ref_audio, sr, segment_length)

    # Используем средний MFCC эталонного файла как опорную точку
    ref_mfccs = [extract_mfcc(seg, sr) for seg in ref_segments]
    reference_mfcc = np.mean(ref_mfccs, axis=0)

    logger.info(f"Загрузка тестового аудиофайла: {test_file}")
    test_audio, _ = librosa.load(test_file, sr=sr)
    test_segments = split_audio(test_audio, sr, segment_length)

    logger.info(f"Вычисление MFCC для {len(test_segments)} тестовых сегментов...")
    test_mfccs = Parallel(n_jobs=-1)(delayed(extract_mfcc)(seg, sr) for seg in test_segments)
    test_mfccs = np.array(test_mfccs)

    logger.info("Загрузка модели...")
    model = load_model(model_path,
                       custom_objects={'contrastive_loss': contrastive_loss, 'cosine_distance': cosine_distance})

    logger.info("Применение модели к тестовому аудио...")
    predictions = predict_with_model(model, test_mfccs, reference_mfcc)

    # Выводим результаты
    for i, score in enumerate(predictions):
        logger.info(f"Сегмент {i}: Уровень совпадения: {score:.4f}")

    # Агрегация результатов
    avg_score = np.mean(predictions)
    logger.info(f"Средний уровень совпадения: {avg_score:.4f}")
    print(f"Средний уровень совпадения голосов: {avg_score:.4f}")


# Функция контрастной потери (определяется повторно для корректного применения модели)
def contrastive_loss(y_true, y_pred, margin=1.0):
    """Контрастная функция потерь для сиамской сети."""
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


# Косинусное расстояние (определяется для использования в модели)
def cosine_distance(vectors):
    """Возвращает косинусное расстояние между двумя векторами."""
    x, y = vectors
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
    return 1 - tf.reduce_sum(x * y, axis=-1)


# Пример вызова функции для демонстрации работы
if __name__ == "__main__":
    # Пути к файлам и модели
    reference_voice = "downloads/raya_references/Lp. Идеальный МИР #30 НОВЫЙ ПЕРСОНАЖ • Майнкрафт.wav"
    test_voice = "E:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #10 ЖИВОЙ РОБОТ • Майнкрафт.wav"
    model_path = "models/voice_recognition_model_v1.keras"

    # Запуск демонстрации
    demo_voice_recognition(model_path, reference_voice, test_voice)
