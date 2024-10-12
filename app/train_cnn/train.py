import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.train_cnn.architecture import build_cnn_model  # Импорт архитектуры CNN модели
from app.utils.logger import get_logger

# Инициализация логгера
logger = get_logger("cnn_training")

# Пути к данным
DATA_FOLDER = "../data/datasets"
MODEL_SAVE_PATH = "../data/models/cnn_voice_model.keras"

# Параметры обучения
LEARNING_RATE = 0.001  # Начальная скорость обучения
BATCH_SIZE = 64  # Размер батча
EPOCHS = 50  # Максимальное количество эпох
PATIENCE = 5  # Количество эпох для ранней остановки
INPUT_SHAPE = (157, 20, 1)  # Размер входного изображения (временные шаги, количество MFCC, 1 канал)
NUM_CLASSES = 1  # Для бинарной классификации (1 класс)

# Настройка сеанса TensorFlow
tf.keras.backend.clear_session()


def load_dataset(file_path):
    """
    Загружает данные из .npz файла и возвращает кортеж из двух элементов: признаки и метки.

    :param file_path: Путь к .npz файлу.
    :return: Два numpy массива — признаки X и метки y.
    """
    logger.info(f"Загрузка данных из {file_path}...")
    data = np.load(file_path)
    X, y = data['X'], data['y']
    logger.info(f"Размер данных: {X.shape}, меток: {y.shape}")
    return X, y


def preprocess_data(X, y):
    """
    Предобработка данных: нормализация и приведение формы.

    :param X: Признаки.
    :param y: Метки.
    :return: Преобразованные признаки и метки.
    """
    # Нормализация входных данных
    X = X / np.max(X)  # Приведение данных к диапазону [0, 1]

    # Преобразование меток в бинарный формат
    if NUM_CLASSES == 1:
        y = y.astype(np.float32)
    else:
        y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    # Изменение формы данных (добавляем ось канала)
    X = X[..., np.newaxis]
    logger.info(f"Данные после предобработки: {X.shape}, метки: {y.shape}")
    return X, y


def plot_metrics(history, output_folder):
    """
    Визуализация процесса обучения: потери и точности.

    :param history: История обучения модели.
    :param output_folder: Папка для сохранения изображений.
    """
    # Построение графиков потерь и точности
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_metrics.png'))
    plt.show()


def train_model(X_train, y_train, X_val, y_val, learning_rate, batch_size, epochs, patience, model_save_path):
    """
    Тренирует модель на обучающих данных и оценивает её на валидационной выборке.

    :param X_train: Признаки обучающей выборки.
    :param y_train: Метки обучающей выборки.
    :param X_val: Признаки валидационной выборки.
    :param y_val: Метки валидационной выборки.
    :param learning_rate: Начальная скорость обучения.
    :param batch_size: Размер батча.
    :param epochs: Количество эпох.
    :param patience: Количество эпох для ранней остановки.
    :param model_save_path: Путь для сохранения обученной модели.
    :return: История обучения.
    """
    logger.info("Инициализация и компиляция CNN модели...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Добавление callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Запуск обучения
    logger.info("Начало обучения модели...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr],
                        verbose=2)
    logger.info("Обучение завершено!")
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Оценивает модель на тестовых данных и выводит метрики.

    :param model: Обученная модель.
    :param X_test: Признаки тестовой выборки.
    :param y_test: Метки тестовой выборки.
    """
    logger.info("Оценка модели на тестовых данных...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    logger.info(f"Тестовая точность: {test_accuracy * 100:.2f}%")

    # Получение предсказаний
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    # Отчёт о классификации
    logger.info("Отчёт о классификации:")
    print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title("Матрица ошибок")
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.show()


if __name__ == "__main__":
    logger.info("Запуск обучения CNN модели для распознавания целевого голоса...")

    # Загрузка данных
    X_train, y_train = load_dataset(os.path.join(DATA_FOLDER, "train_data.npz"))
    X_val, y_val = load_dataset(os.path.join(DATA_FOLDER, "validation_data.npz"))
    X_test, y_test = load_dataset(os.path.join(DATA_FOLDER, "test_data.npz"))

    # Предобработка данных
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Обучение модели
    model, history = train_model(X_train, y_train, X_val, y_val,
                                 learning_rate=LEARNING_RATE,
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS,
                                 patience=PATIENCE,
                                 model_save_path=MODEL_SAVE_PATH)

    # Визуализация метрик обучения
    plot_metrics(history, DATA_FOLDER)

    # Оценка модели
    evaluate_model(model, X_test, y_test)
