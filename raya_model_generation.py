import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Пути к данным
BASE_DIR = os.path.dirname(__file__)
RAYA_VOICE_PATH = os.path.join(BASE_DIR, "downloads", "raya_references")
OTHER_VOICES_PATH = os.path.join(BASE_DIR, "downloads", "other_references")
NOISES_PATH = os.path.join(BASE_DIR, "downloads", "noises")


# Функция загрузки всех аудиофайлов в папке
def load_all_files(directory):
    """Загружает все аудиофайлы из указанной директории."""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            files.append(os.path.join(directory, filename))
    return files


# Генерация фрагментов аудио
def extract_audio_segments(file_list, sr=16000, segment_length=2):
    """Разделяет аудиофайлы на сегменты фиксированной длины."""
    step = int(segment_length * sr)
    segments = []
    for file in file_list:
        audio, _ = librosa.load(file, sr=sr)
        for start in range(0, len(audio) - step, step // 2):  # Перекрытие на 50%
            segment = audio[start:start + step]
            if len(segment) == step:
                segments.append(segment)
    return segments


# Функция нормализации количества сегментов путем дублирования
def normalize_segments(raya_segments, other_segments, noise_segments):
    """Нормализует количество сегментов путем дублирования."""
    max_len = max(len(raya_segments), len(other_segments), len(noise_segments))

    # Функция дублирования сегментов
    def duplicate_segments(segments, target_length):
        while len(segments) < target_length:
            segments.extend(segments[:target_length - len(segments)])
        return segments[:target_length]

    # Нормализация количества сегментов
    raya_segments = duplicate_segments(raya_segments, max_len)
    other_segments = duplicate_segments(other_segments, max_len)
    noise_segments = duplicate_segments(noise_segments, max_len)

    return raya_segments, other_segments, noise_segments


# Функция подготовки данных
def prepare_data(raya_files, other_files, noise_files, segment_length=2, sr=16000):
    """Формирует пары данных для обучения."""
    # Извлечение сегментов голосов и шумов
    raya_segments = extract_audio_segments(raya_files, sr, segment_length)
    other_segments = extract_audio_segments(other_files, sr, segment_length)
    noise_segments = extract_audio_segments(noise_files, sr, segment_length)

    # Отображение количества сегментов
    print(f"Количество сегментов голосов Райи: {len(raya_segments)}")
    print(f"Количество сегментов других голосов: {len(other_segments)}")
    print(f"Количество сегментов шумов: {len(noise_segments)}")

    # Нормализация количества сегментов
    raya_segments, other_segments, noise_segments = normalize_segments(raya_segments, other_segments, noise_segments)

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

    # Шумовые пары (голос Райи — шум)
    for seg_raya, seg_noise in zip(raya_segments, noise_segments):
        x1.append(seg_raya)
        x2.append(seg_noise)
        y.append(0)

    # Преобразование в MFCC с нормализацией (добавлено нормирование)
    x1_mfcc = [(librosa.feature.mfcc(y=s, sr=sr, n_mfcc=13).T[:50] - np.mean(s)) / (np.std(s) + 1e-10) for s in x1]
    x2_mfcc = [(librosa.feature.mfcc(y=s, sr=sr, n_mfcc=13).T[:50] - np.mean(s)) / (np.std(s) + 1e-10) for s in x2]

    return np.array(x1_mfcc), np.array(x2_mfcc), np.array(y)


# Создание архитектуры базовой сети
def create_base_network(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(512, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    return model


# Контрастная функция потерь для обучения SNN
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def main():
    # Подготовка данных
    raya_files = load_all_files(RAYA_VOICE_PATH)
    other_files = load_all_files(OTHER_VOICES_PATH)
    noise_files = load_all_files(NOISES_PATH)

    # Генерация данных
    x1, x2, y = prepare_data(raya_files, other_files, noise_files)

    # Разделение на обучающую и тестовую выборку
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=42)

    # Архитектура сети
    input_shape = x1_train.shape[1:]
    base_network = create_base_network(input_shape)

    # Входные данные для двух потоков
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Извлечение признаков
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Вычисление евклидова расстояния
    distance = layers.Lambda(lambda tensors: tf.norm(tensors[0] - tensors[1], axis=1))([processed_a, processed_b])

    # Определение модели
    model = models.Model([input_a, input_b], distance)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=contrastive_loss)

    # Обучение модели
    model.fit([x1_train, x2_train], y_train, batch_size=32, epochs=30, validation_split=0.2)

    # Тестирование модели
    y_pred = model.predict([x1_test, x2_test])
    y_pred_labels = (y_pred < 0.5).astype(int)

    # Построение матрицы ошибок
    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Разные голоса', 'Голос Райи'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Матрица ошибок для модели распознавания голоса")
    plt.show()

    # Сохранение модели
    model.save('models/voice_recognition_model.keras')
    print("Модель успешно сохранена как voice_recognition_model.keras")


if __name__ == "__main__":
    main()
