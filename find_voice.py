import librosa
import numpy as np
import tensorflow as tf

from utils.my_logger import logger
from voice_model_generation.model import cosine_distance
from voice_model_generation.preprocessing import extract_audio_segments
from voice_model_generation.train_test import contrastive_loss


def load_audio_file(file_path, target_sr=16000):
    """Загружает аудиофайл и возвращает его в виде массива."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        logger.info(f"Файл {file_path} успешно загружен с частотой дискретизации {sr}.")
        return audio
    except Exception as e:
        logger.error(f"Ошибка при загрузке аудиофайла {file_path}: {e}")
        return None


def demo_voice_recognition(model_path, reference_mfcc_path, test_voice_path, batch_size=32):
    # Загрузка модели
    logger.info(f"Загрузка модели: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={'cosine_distance': cosine_distance,
                                                                   "contrastive_loss": contrastive_loss})

    # Загрузка эталонных MFCC
    logger.info(f"Загрузка эталонных MFCC из {reference_mfcc_path}")
    reference_mfcc = np.load(reference_mfcc_path)
    reference_mfcc = np.expand_dims(reference_mfcc, axis=0)  # Добавляем размерность батча

    # Загрузка тестового аудиофайла
    logger.info(f"Загрузка тестового аудиофайла: {test_voice_path}")
    test_audio = load_audio_file(test_voice_path, target_sr=16000)
    if test_audio is None:
        logger.error(f"Не удалось загрузить аудиофайл: {test_voice_path}")
        return

    # Извлечение аудиосегментов из загруженного аудиомассива
    test_segments = extract_audio_segments([test_audio], sr=16000, segment_length=2)
    if len(test_segments) == 0:
        logger.error(
            "Не удалось извлечь сегменты из тестового аудио. Проверьте параметры сегментации или содержание аудиофайла.")
        return

    logger.info(f"Общее количество извлеченных сегментов: {len(test_segments)}")

    # Преобразование каждого сегмента в MFCC
    test_mfccs = np.array([librosa.feature.mfcc(y=segment, sr=16000, n_mfcc=13).T for segment in test_segments])

    # Проверка на размерности MFCC
    if reference_mfcc.shape[1] != test_mfccs.shape[1]:
        logger.warning(f"Несоответствие размеров MFCC. Эталон: {reference_mfcc.shape[1]}, Тест: {test_mfccs.shape[1]}")
        return

    # Итеративное предсказание по сегментам
    logger.info(f"Применение модели для {len(test_mfccs)} сегментов...")
    scores = []

    # Применяем предсказание блоками (батчами) для экономии памяти
    for start in range(0, len(test_mfccs), batch_size):
        end = min(start + batch_size, len(test_mfccs))
        test_batch = test_mfccs[start:end]

        # Повторяем эталонный MFCC для каждого сегмента в текущем батче
        reference_batch = np.repeat(reference_mfcc, len(test_batch), axis=0)

        # Выполняем предсказание для текущего батча
        batch_scores = model.predict([reference_batch, test_batch], verbose=0)
        scores.extend(batch_scores)

        logger.info(f"Обработано сегментов: {end}/{len(test_mfccs)}")

    # Вывод результатов для каждого сегмента
    for i, score in enumerate(scores):
        logger.info(f"Сегмент {i}: Уровень совпадения: {score[0]:.4f}")


# Запуск демо-функции
if __name__ == "__main__":
    model_path = "models/voice_recognition_model_v1.keras"
    reference_mfcc_path = "models/reference_mfcc.npy"
    test_voice_path = "E:/4K_Video_Downloader/Lp  Идеальный Мир · Майнкрафт/Lp. Идеальный МИР #10 ЖИВОЙ РОБОТ • Майнкрафт.wav"
    demo_voice_recognition(model_path, reference_mfcc_path, test_voice_path)
