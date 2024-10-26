# config.py

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
    SAMPLE_RATE = 44100
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = SPECTROGRAM_SIZE[0]

    # Параметры аугментации
    AUGMENTATION_THRESHOLD = 0.8  # Порог дисбаланса для запуска аугментации
    AUGMENTATION_RATIO = 0.5  # Отношение количества аугментированных данных к существующим
    MAX_AUGMENTATION_PROCESSES = 4  # Количество параллельных процессов для аугментации
    NOISE_FACTOR = 0.005  # Фактор добавляемого шума для аугментации

    # Параметры обучения
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    USE_CUDA = False
