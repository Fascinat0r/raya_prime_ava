import tensorflow as tf
from tensorflow.keras import layers


# Создание архитектуры базовой сети
def create_base_network(input_shape):
    """Создает и возвращает базовую архитектуру нейросети."""
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(128, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(512, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])
    return model


# Функция для вычисления косинусного расстояния
def cosine_distance(vectors):
    """Возвращает косинусное расстояние между двумя векторами."""
    x, y = vectors
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
    return 1 - tf.reduce_sum(x * y, axis=-1)


def create_siamese_network(input_shape):
    """Создает и возвращает модель сиамской сети с косинусным расстоянием."""
    base_network = create_base_network(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Заменяем на косинусное расстояние
    distance = layers.Lambda(cosine_distance)([processed_a, processed_b])

    model = tf.keras.Model([input_a, input_b], distance)
    return model
