import tensorflow as tf
from tensorflow.keras import layers, Model


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


# Создание сиамской сети без кастомных функций
def create_siamese_network(input_shape):
    """Создает и возвращает модель сиамской сети с косинусным расстоянием."""
    base_network = create_base_network(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Используем встроенный слой Dot с параметром normalize=True для косинусного расстояния
    distance = layers.Dot(axes=1, normalize=True)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model
