import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model


def build_cnn_model(input_shape, num_classes):
    """
    Создаёт свёрточную нейронную сеть для распознавания целевого голоса.

    :param input_shape: Размер входных данных (временные шаги, частотные компоненты, 1 канал).
    :param num_classes: Количество выходных классов (1 для бинарной классификации).
    :return: Построенная модель.
    """
    inputs = Input(shape=input_shape, name='input_layer')

    # Свёрточные слои с Batch Normalization и MaxPooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(x)

    # Преобразование в вектор
    x = Flatten(name='flatten')(x)

    # Полносвязные слои с Dropout
    x = Dense(256, activation='relu', name='fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc_2')(x)

    # Выходной слой
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid', name='output_layer')(x)  # Для бинарной классификации
    else:
        outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)  # Для многоклассовой классификации

    # Создание модели
    model = Model(inputs, outputs, name='cnn_voice_classifier')
    return model
