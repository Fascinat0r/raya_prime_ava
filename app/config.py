# config.py
# Description: Конфигурационные параметры для проекта.
import os


class Config:
    # Путь к папке с данными
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

    # Путь к папке с метаданными
    METADATA_PATH = os.path.join(DATA_PATH, "metadata.csv")

    # Путь к папке с разделенными данными
    SPLIT_DATA_PATH = os.path.join(DATA_PATH, "split_data")

    # Путь к папке с тренировочными данными
    TRAIN_DATA_PATH = os.path.join(SPLIT_DATA_PATH, "train_metadata.csv")

    # Путь к папке с тестовыми данными
    TEST_DATA_PATH = os.path.join(SPLIT_DATA_PATH, "test_metadata.csv")

    # Путь к папке с сырыми данными
    RAW_FOLDER = os.path.join(DATA_PATH, "raw")

    # Путь к папке с предсказаниями
    PREDICTIONS_PATH = os.path.join(DATA_PATH, "predictions")

    # Путь к папке с аудиофайлами
    AUDIO_PATH = os.path.join(DATA_PATH, "audio")

    # Путь к папке с мел-спектрограммами
    SPECTROGRAMS_PATH = os.path.join(DATA_PATH, "spectrograms")

    # Путь для сохранения модели
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "train", "weights")

    # Размер спектрограммы
    SPECTROGRAM_SIZE = (64, 64)

    # Параметры для обработки аудио
    MAX_PREPROCESSING_PROCESSES = 4
    SAMPLE_RATE = 44100
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = SPECTROGRAM_SIZE[0]

    # Параметры аугментации
    AUGMENTATION_THRESHOLD = 0.8  # Порог дисбаланса для запуска аугментации
    AUGMENTATION_RATIO = 0.75  # Отношение количества аугментированных данных к существующим
    MAX_AUGMENTATION_PROCESSES = 4  # Количество параллельных процессов для аугментации
    NOISE_FACTOR_RANGE = [0.005, 0.1]  # Фактор добавляемого шума для аугментации]
    VOLUME_RANGE = [0.005, 0.01]

    # Параметры для предсказания
    PREDICTION_THRESHOLD = 0.6
    MIN_VOICE_DURATION = 0.5
    MAX_PAUSE = 1
    WINDOW_SIZE = 3

    # Параметры обучения
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    USE_CUDA = False

    # Параметры постобработки
    THRESHOLD = 0.6  # Порог вероятности для классификации как голос
    MIN_DURATION = 0.5  # Минимальная продолжительность интервала в секундах
    MERGE_GAP = 0.1  # Максимальный промежуток между интервалами для объединения
    # MEDIAN_KERNEL_SIZE = 3  # Размер ядра медианного фильтра
    MOVING_AVERAGE_WINDOW = 6  # Размер окна для скользящего среднего
