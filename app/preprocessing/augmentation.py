import random

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


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
    return librosa.effects.time_stretch(audio_data, rate)


def add_noise(audio_data, noise_factor_range=(0.002, 0.01)):
    """
    Добавление белого шума.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param noise_factor_range: Диапазон уровней шума.
    """
    noise_factor = random.uniform(noise_factor_range[0], noise_factor_range[1])
    noise = np.random.randn(len(audio_data))
    return audio_data + noise_factor * noise


def change_volume(audio_data, sample_rate, gain_range=(-5, 5)):
    """
    Изменение громкости.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param sample_rate: Частота дискретизации.
    :param gain_range: Диапазон изменения громкости в dB.
    """
    gain = random.randint(gain_range[0], gain_range[1])
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    # Применение изменения громкости
    altered_segment = audio_segment + gain
    return np.array(altered_segment.get_array_of_samples())


def apply_reverb(audio_data, sample_rate, reverb_intensity_range=(200, 500)):
    """
    Применение реверберации.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param sample_rate: Частота дискретизации.
    :param reverb_intensity_range: Диапазон частот среза для имитации реверберации.
    """
    reverb_intensity = random.randint(reverb_intensity_range[0], reverb_intensity_range[1])
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    # Применение реверберации через фильтр
    reverb_audio = audio_segment.low_pass_filter(reverb_intensity)
    return np.array(reverb_audio.get_array_of_samples())


def high_pass_filter(audio_data, cutoff_range=(200, 1000), sample_rate=16000):
    """
    Применение высокочастотного фильтра.
    :param audio_data: Аудиоданные в виде массива numpy.
    :param cutoff_range: Диапазон порога частоты (в Гц), ниже которого частоты будут подавляться.
    :param sample_rate: Частота дискретизации.
    """
    cutoff = random.randint(cutoff_range[0], cutoff_range[1])
    return librosa.effects.preemphasis(audio_data, coef=cutoff / sample_rate)


def augment_file(input_file, output_file):
    """
    Применяет случайную аугментацию к входному аудиофайлу и сохраняет результат.
    :param input_file: Путь к исходному .wav файлу.
    :param output_file: Путь для сохранения аугментированного .wav файла.
    """
    print(f"Обработка файла: {input_file}")

    # Загрузка аудиофайла
    audio_data, sample_rate = librosa.load(input_file, sr=None)

    # Список всех доступных методов аугментации
    augmentation_methods = [
        lambda x: pitch_shift(x, sample_rate, n_steps_range=(-1, 1)),
        lambda x: time_stretch(x, rate_range=(0.8, 1.2)),
        lambda x: add_noise(x, noise_factor_range=(0.002, 0.02)),
        lambda x: change_volume(x, sample_rate, gain_range=(-10, 10)),
        lambda x: apply_reverb(x, sample_rate, reverb_intensity_range=(200, 800)),
        lambda x: high_pass_filter(x, cutoff_range=(200, 2000), sample_rate=sample_rate)
    ]

    # Выбор случайной аугментации
    selected_augmentation_idx = random.randint(0, len(augmentation_methods) - 1)
    selected_augmentation = augmentation_methods[selected_augmentation_idx]

    # Названия методов аугментации на русском
    augmentation_methods_names = [
        "Изменение высоты тона",
        "Изменение скорости",
        "Добавление шума",
        "Изменение громкости",
        "Применение реверберации",
        "Применение высокочастотного фильтра"
    ]

    print(f"Выбрана аугментация: {augmentation_methods_names[selected_augmentation_idx]}")

    # Применение выбранной аугментации
    augmented_data = selected_augmentation(audio_data)

    # Сохранение результата
    sf.write(output_file, augmented_data, sample_rate)
    print(f"Аугментированный файл сохранен: {output_file}")


# Пример использования
input_file_path = "data/raw/0__happiness_neutral_h_040.wav"
output_file_path = "data/augmented/example_augmented.wav"

# Применение случайной аугментации
augment_file(input_file_path, output_file_path)
