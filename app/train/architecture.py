import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model


def build_base_network(input_shape):
    """
    Создает архитектуру базовой сети для сиамской сети.

    :param input_shape: Размер входного изображения (например, (n_mfcc, временные окна, 1)).
    :return: Модель базовой сети.
    """
    inputs = Input(shape=input_shape, name='input_layer')

    # Сверточные слои с уменьшением размерности через MaxPooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(x)

    # Преобразование в вектор через Flatten
    x = Flatten(name='flatten')(x)

    # Полносвязанные слои для извлечения более абстрактных признаков
    x = Dense(256, activation='relu', name='fc_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)  # Dropout для регуляризации

    x = Dense(128, activation='relu', name='fc_2')(x)
    return Model(inputs, x, name='base_network')


def euclidean_distance(vectors):
    """
    Вычисление евклидова расстояния между двумя векторами с использованием tf.norm.

    :param vectors: Список из двух входных векторов.
    :return: Евклидово расстояние между двумя входами.
    """
    (featA, featB) = vectors
    return tf.norm(featA - featB, ord='euclidean', axis=1, keepdims=True)


def cosine_similarity(vectors):
    """
    Вычисление косинусного сходства между двумя векторами.

    :param vectors: Список из двух входных векторов.
    :return: Косинусное сходство между двумя входами.
    """
    (featA, featB) = vectors
    return tf.keras.losses.cosine_similarity(featA, featB, axis=1)


def build_siamese_network(input_shape, distance_metric='euclidean'):
    """
    Создает сиамскую сеть на основе базовой сети с возможностью выбора метрики расстояния.

    :param input_shape: Размер входного изображения (например, (n_mfcc, временные окна, 1)).
    :param distance_metric: Метрика расстояния: 'euclidean' или 'cosine'.
    :return: Полная модель сиамской сети.
    """
    # Создание базовой сети
    base_network = build_base_network(input_shape)

    # Входные данные для двух ветвей сети
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')

    # Получение эмбеддингов для каждой ветви
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Выбор метрики расстояния
    if distance_metric == 'euclidean':
        distance = Lambda(euclidean_distance, name='euclidean_distance')([processed_a, processed_b])
    elif distance_metric == 'cosine':
        distance = Lambda(cosine_similarity, name='cosine_similarity')([processed_a, processed_b])
    else:
        raise ValueError("Метрика расстояния должна быть 'euclidean' или 'cosine'")

    # Создание модели сиамской сети с двумя входами и одним выходом (расстояние или сходство)
    model = Model([input_a, input_b], distance, name='siamese_network')

    return model


if __name__ == "__main__":
    # Пример инициализации сети
    input_shape = (20, 200, 1)  # Размер входного изображения (например, 20 MFCC коэффициентов на 200 временных окон)

    # Сиамская модель с евклидовым расстоянием
    siamese_model = build_siamese_network(input_shape, distance_metric='euclidean')
    print("Сиамская модель с евклидовым расстоянием:")
    siamese_model.summary()

    # Сиамская модель с косинусным сходством
    siamese_model_cosine = build_siamese_network(input_shape, distance_metric='cosine')
    print("\nСиамская модель с косинусным сходством:")
    siamese_model_cosine.summary()
