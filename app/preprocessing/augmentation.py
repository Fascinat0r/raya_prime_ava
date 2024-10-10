import os
import random

import librosa
import numpy as np
import soundfile as sf

from app.utils.logger import get_logger

logger = get_logger("augmentation")


def pitch_shift(audio_data, sample_rate, n_steps_range=(-3, 3)):
    """
    Изменение высоты тона.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param sample_rate: Частота дискретизации.
    :param n_steps_range: Диапазон изменения высоты тона (полутона).
    """
    n_steps = random.randint(n_steps_range[0], n_steps_range[1])
    return librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)


def time_stretch(audio_data, rate_range=(0.8, 1.5)):
    """
    Изменение скорости воспроизведения.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param rate_range: Диапазон скорости (значения >1 — ускорение, <1 — замедление).
    """
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio_data, rate=rate)


def generate_pink_noise(length: int):
    """
    Генерация розового шума.
    :param length: Длина массива, равная количеству сэмплов в аудиофайле.
    :return: Массив numpy с розовым шумом.
    """
    uneven = length % 2
    X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # Убывание частотной мощности
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return y[:length]


def add_noise(audio_data, sample_rate, noise_type='white', noise_factor_range=(0.002, 0.01),
              noise_files_folder="../data/noises"):
    """
    Добавление различных видов шума.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param sample_rate: Частота дискретизации.
    :param noise_type: Тип шума ('white', 'pink', 'impulse', 'file').
    :param noise_factor_range: Диапазон уровней шума.
    :param noise_files_folder: Папка с аудиофайлами, содержащими шумы.
    """
    noise_factor = random.uniform(noise_factor_range[0], noise_factor_range[1])

    if noise_type == 'white':
        # Белый шум
        noise = np.random.randn(len(audio_data))
    elif noise_type == 'pink':
        # Розовый шум
        noise = generate_pink_noise(len(audio_data))
    elif noise_type == 'impulse':
        # Импульсный шум
        noise = np.zeros(len(audio_data))
        impulse_positions = np.random.randint(0, len(audio_data), size=10)
        noise[impulse_positions] = np.random.randn(10) * noise_factor
    elif noise_type == 'file':
        # Выбираем случайный шумовой файл из папки
        noise_files = [f for f in os.listdir(noise_files_folder) if f.endswith(('.wav', '.mp3'))]
        if not noise_files:
            raise ValueError(f"Не найдено файлов в папке {noise_files_folder}.")

        random_noise_file = os.path.join(noise_files_folder, random.choice(noise_files))

        # Загрузка шума из файла и подгонка длины
        noise, _ = librosa.load(random_noise_file, sr=sample_rate)
        if len(noise) > len(audio_data):
            # Если шум длиннее, берем случайный фрагмент той же длины
            start_idx = random.randint(0, len(noise) - len(audio_data))
            noise = noise[start_idx:start_idx + len(audio_data)]
        else:
            # Если шум короче, повторяем до нужной длины
            noise = np.tile(noise, int(np.ceil(len(audio_data) / len(noise))))[:len(audio_data)]
    else:
        raise ValueError(f"Неизвестный тип шума: {noise_type}")

    # Масштабируем шум в зависимости от мощности оригинального аудиосигнала
    rms_audio = np.sqrt(np.mean(audio_data ** 2))
    rms_noise = np.sqrt(np.mean(noise ** 2))
    scaled_noise = noise * (rms_audio / (rms_noise + 1e-6)) * noise_factor

    return audio_data + scaled_noise


def high_pass_filter(audio_data, cutoff_range=(200, 1000), sample_rate=16000):
    """
    Применение высокочастотного фильтра.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param cutoff_range: Диапазон порога частоты (в Гц), ниже которого частоты будут подавляться.
    :param sample_rate: Частота дискретизации.
    """
    cutoff = random.randint(cutoff_range[0], cutoff_range[1])
    return librosa.effects.preemphasis(audio_data, coef=cutoff / sample_rate)


def augment_file(input_file, output_file, augmentation_probs=None):
    """
    Применяет случайную аугментацию к входному аудиофайлу с учетом вероятностей и сохраняет результат.
    :param input_file: Путь к исходному .wav файлу.
    :param output_file: Путь для сохранения аугментированного .wav файла.
    :param augmentation_probs: Словарь с шансами выбора каждого типа аугментации.
    """
    if augmentation_probs is None:
        augmentation_probs = [0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.1]

    logger.info(f"Обработка файла: {input_file}")

    # Загрузка аудиофайла
    audio_data, sample_rate = librosa.load(input_file, sr=None)

    # Список всех доступных методов аугментации и их вероятности
    augmentation_methods = [
        lambda x: pitch_shift(x, sample_rate, n_steps_range=(-1, 1)),
        lambda x: time_stretch(x, rate_range=(0.8, 1.2)),
        lambda x: add_noise(x, sample_rate, noise_type='white', noise_factor_range=(0.002, 0.02)),
        lambda x: add_noise(x, sample_rate, noise_type='pink', noise_factor_range=(0.002, 0.02)),
        lambda x: add_noise(x, sample_rate, noise_type='impulse', noise_factor_range=(0.002, 0.02)),
        lambda x: add_noise(x, sample_rate, noise_type='file', noise_factor_range=(0.005, 0.10)),
        lambda x: high_pass_filter(x, cutoff_range=(200, 2000), sample_rate=sample_rate)
    ]

    # Названия методов аугментации на русском
    augmentation_methods_names = [
        "Изменение высоты тона",
        "Изменение скорости",
        "Добавление белого шума",
        "Добавление розового шума",
        "Добавление импульсного шума",
        "Добавление шума из файла",
        "Применение высокочастотного фильтра"
    ]

    # Выбор случайной аугментации на основе вероятностей
    selected_augmentation_idx = random.choices(
        population=range(len(augmentation_methods)),
        weights=augmentation_probs,
        k=1
    )[0]
    selected_augmentation = augmentation_methods[selected_augmentation_idx]

    logger.info(f"Выбрана аугментация: {augmentation_methods_names[selected_augmentation_idx]}")

    # Применение выбранной аугментации
    augmented_data = selected_augmentation(audio_data)

    # Сохранение результата
    sf.write(output_file, augmented_data, sample_rate)
    logger.info(f"Аугментированный файл сохранен: {output_file}")


if __name__ == "__main__":
    # Пример использования
    augment_file("../data/segments/0_0001.wav", "../data/segments/0_0001_augmented.wav")
    augment_file("../data/segments/1_0001.wav", "../data/segments/1_0001_augmented.wav")
