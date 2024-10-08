import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from utils.my_logger import logger
from voice_model_generation.data_loader import load_all_files
from voice_model_generation.model import create_siamese_network
from voice_model_generation.preprocessing import extract_audio_segments, normalize_segments
from voice_model_generation.train_test import plot_confusion_matrix, get_early_stopping_callback

# Пути к данным
BASE_DIR = os.path.dirname(__file__)
RAYA_VOICE_PATH = os.path.join(BASE_DIR, "downloads", "raya_references")
OTHER_VOICES_PATH = os.path.join(BASE_DIR, "downloads", "other_references")
NOISES_PATH = os.path.join(BASE_DIR, "downloads", "noises")


def pad_or_trim(segment, target_length):
    """
    Обрезает или дополняет аудиосегмент до указанной длины.

    Args:
        segment (np.array): Аудиосегмент.
        target_length (int): Целевая длина сегмента.

    Returns:
        np.array: Сегмент, обрезанный или дополненный до нужной длины.
    """
    if len(segment) > target_length:
        return segment[:target_length]
    elif len(segment) < target_length:
        padding = np.zeros(target_length - len(segment))
        return np.concatenate([segment, padding])
    else:
        return segment


def prepare_data(raya_files, other_files, noise_files, segment_length=2, sr=16000):
    """
    Формирует пары данных для обучения.
    """
    logger.info("Подготовка данных...")

    # Извлечение загруженных аудиоданных из кортежей
    raya_audio, _ = raya_files
    other_audio, _ = other_files
    noise_audio, _ = noise_files

    # Извлечение сегментов из каждого типа данных
    raya_segments = extract_audio_segments(raya_audio, sr, segment_length)
    other_segments = extract_audio_segments(other_audio, sr, segment_length)
    other_segments += extract_audio_segments(noise_audio, sr, segment_length)

    # Нормализация количества сегментов
    raya_segments, other_segments = normalize_segments(raya_segments, other_segments)

    # Обрезка или дополнение каждого сегмента до одинаковой длины
    step = int(segment_length * sr)
    raya_segments = [pad_or_trim(seg, step) for seg in raya_segments]
    other_segments = [pad_or_trim(seg, step) for seg in other_segments]

    x1, x2, y = [], [], []

    # Положительные пары (голос Райи — голос Райи)
    for seg in raya_segments:
        x1.append(seg)
        x2.append(seg)
        y.append(1)

    # Отрицательные пары (голос Райи — другой голос)
    for seg_raya, seg_other in zip(raya_segments, other_segments):
        x1.append(seg_raya)
        x2.append(seg_other)
        y.append(0)

    logger.info(f"Положительных пар: {y.count(1)}, Отрицательных пар: {y.count(0)}")
    logger.info(f"Общее количество пар: {len(x1)}")

    # Преобразование в numpy-массивы с одинаковой длиной
    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return x1, x2, y


def voice_model_generation():
    logger.info("Запуск основного процесса создания модели распознавания голоса...")

    # Загрузка данных
    raya_files = load_all_files(RAYA_VOICE_PATH)
    other_files = load_all_files(OTHER_VOICES_PATH)
    noise_files = load_all_files(NOISES_PATH)

    # Генерация данных
    x1, x2, y = prepare_data(raya_files, other_files, noise_files)

    # Преобразование в MFCC
    logger.info("Преобразование аудиосегментов в MFCC...")
    sr = 16000
    x1_mfcc = [librosa.feature.mfcc(y=s, sr=sr, n_mfcc=13).T for s in x1]
    x2_mfcc = [librosa.feature.mfcc(y=s, sr=sr, n_mfcc=13).T for s in x2]

    # Нормализация MFCC
    x1_mfcc = [(mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10) for mfcc in x1_mfcc]
    x2_mfcc = [(mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10) for mfcc in x2_mfcc]

    # Преобразование в массивы numpy
    x1_mfcc = np.array(x1_mfcc)
    x2_mfcc = np.array(x2_mfcc)

    logger.info(f"Общая размерность данных: {len(x1_mfcc)} пар")

    # Разделение данных на обучающую и тестовую выборку
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_mfcc, x2_mfcc, y, test_size=0.2,
                                                                             random_state=42)
    np.save("models/reference_mfcc.npy", x1_train)
    # Создание архитектуры сиамской сети
    input_shape = x1_train.shape[1:]
    model = create_siamese_network(input_shape)

    # Компиляция модели
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    # Обучение модели с EarlyStopping
    logger.info("Начало обучения модели...")
    early_stopping = get_early_stopping_callback(patience=5)
    model.fit([x1_train, x2_train], y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping])

    # Тестирование модели
    logger.info("Проверка модели...")
    y_pred = model.predict([x1_test, x2_test])
    y_pred_labels = (y_pred < 0.5).astype(int)

    # Построение матрицы ошибок
    logger.info("Построение матрицы ошибок...")
    plot_confusion_matrix(y_test, y_pred_labels)

    # Сохранение модели
    logger.info("Сохранение модели...")
    model.save('models/voice_recognition_model_v2.keras')
    logger.info("Модель успешно сохранена как voice_recognition_model_v2.keras")


if __name__ == "__main__":
    voice_model_generation()
