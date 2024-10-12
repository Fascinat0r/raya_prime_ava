import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from app.train.architecture import build_siamese_network
from app.utils.logger import get_logger

# Инициализация логгера
logger = get_logger("training")

# Пути к данным
TRAIN_DATA_FILE = "../data/train/train_pairs.npz"
VALIDATION_DATA_FILE = "../data/train/validation_pairs.npz"

# Параметры обучения
LEARNING_RATE = 0.001  # Начальная скорость обучения
BATCH_SIZE = 32  # Размер батча
EPOCHS = 50  # Максимальное количество эпох
PATIENCE = 5  # Количество эпох без улучшений для ранней остановки
MODEL_SAVE_PATH = "../data/models/siamese_model.keras"  # Путь для сохранения обученной модели

# Размер входных данных для модели
INPUT_SHAPE = (157, 20, 1)  # (временные шаги, количество MFCC, 1 канал)


@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Контрастная функция потерь.
    :param y_true: Метки классов (1 — положительные пары, 0 — отрицательные).
    :param y_pred: Расстояние, предсказанное моделью.
    :param margin: Маржинальное значение для отрицательных пар.
    :return: Значение контрастной функции потерь.
    """
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def load_data(file_path):
    """
    Загружает обучающие или валидационные данные из .npz файла.

    :param file_path: Путь к файлу данных.
    :return: Кортеж из трех элементов (X1, X2, y).
    """
    logger.info(f"Загрузка данных из {file_path}...")
    data = np.load(file_path)
    X1, X2, y = data['X1'], data['X2'], data['y']

    # Добавление оси канала, если её нет
    if len(X1.shape) == 3:
        X1 = np.expand_dims(X1, axis=-1)
        X2 = np.expand_dims(X2, axis=-1)

    return X1, X2, y


def create_tf_dataset(X1, X2, y, batch_size):
    """
    Создает tf.data.Dataset из массивов numpy.

    :param X1: Первый элемент пары.
    :param X2: Второй элемент пары.
    :param y: Метки классов.
    :param batch_size: Размер батча.
    :return: Объект tf.data.Dataset.
    """
    logger.info("Создание tf.data.Dataset...")

    # Корректное создание Dataset с нужной структурой
    dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train_siamese_model(train_data, validation_data, learning_rate, batch_size, epochs, patience, model_save_path):
    """
    Обучает сиамскую модель на тренировочном и валидационном наборе.

    :param train_data: Обучающий набор данных (tf.data.Dataset).
    :param validation_data: Валидационный набор данных (tf.data.Dataset).
    :param learning_rate: Начальная скорость обучения.
    :param batch_size: Размер батча.
    :param epochs: Максимальное количество эпох.
    :param patience: Количество эпох без улучшений для ранней остановки.
    :param model_save_path: Путь для сохранения обученной модели.
    """
    logger.info("Инициализация сиамской сети...")
    siamese_network = build_siamese_network(input_shape=INPUT_SHAPE, distance_metric='euclidean')

    # Компиляция модели с контрастной функцией потерь
    siamese_network.compile(optimizer=Adam(learning_rate=learning_rate), loss=contrastive_loss, metrics=['accuracy'])
    logger.info("Модель с контрастной функцией потерь успешно скомпилирована!")

    # Callbacks для мониторинга
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)

    # Запуск обучения
    logger.info("Начало обучения модели...")
    history = siamese_network.fit(train_data,
                                  validation_data=validation_data,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=[early_stopping, model_checkpoint],
                                  verbose=2)
    logger.info("Обучение завершено!")
    return history


if __name__ == "__main__":
    logger.info("Запуск обучения сиамской сети...")

    # Загрузка тренировочных и валидационных данных
    X1_train, X2_train, y_train = load_data(TRAIN_DATA_FILE)
    X1_val, X2_val, y_val = load_data(VALIDATION_DATA_FILE)

    # Создание обучающего и валидационного наборов
    train_dataset = create_tf_dataset(X1_train, X2_train, y_train, BATCH_SIZE)
    validation_dataset = create_tf_dataset(X1_val, X2_val, y_val, BATCH_SIZE)

    # Обучение модели
    train_siamese_model(train_dataset,
                        validation_dataset,
                        learning_rate=LEARNING_RATE,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        patience=PATIENCE,
                        model_save_path=MODEL_SAVE_PATH)
    logger.info(f"Модель сохранена в {MODEL_SAVE_PATH}.")
