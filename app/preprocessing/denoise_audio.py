import logging

import noisereduce as nr
import numpy as np

logger = logging.getLogger("denoise_audio")


def denoise_audio_data(audio_data: np.ndarray, sample_rate: int, chunk_size: int = 50000, prop_decrease: float = 0.8,
                       stationary: bool = False, n_std_thresh_stationary: float = 1.0) -> np.ndarray:
    """
    Удаляет шум из аудиоданных, обрабатывая их сегментами и преобразуя многоканальные данные в моно.

    Аргументы:
    audio_data (np.ndarray): Аудиоданные в виде массива numpy.
    sample_rate (int): Частота дискретизации.
    chunk_size (int): Размер сегмента для обработки (по умолчанию 50000 выборок).
    prop_decrease (float): Уровень подавления шума (по умолчанию 0.8, максимальное значение 1.0).
    stationary (bool): Использовать ли стационарный шумоподавитель (False для спектрального подавления).
    n_std_thresh_stationary (float): Порог стандартного отклонения для определения шума.

    Возвращает:
    np.ndarray: Очищенные аудиоданные.
    """
    logger.info("Запуск шумоподавления аудиоданных...")

    # Проверка количества каналов и преобразование в моно
    if len(audio_data.shape) > 1:
        logger.info("Аудиоданные многоканальные, преобразование в моно.")
        audio_data = convert_to_mono(audio_data)

    # Применение шумоподавления сегментами
    logger.info("Обработка аудиоданных и удаление шума...")
    reduced_noise = reduce_noise_segmented(audio_data, sample_rate, chunk_size, prop_decrease, stationary,
                                           n_std_thresh_stationary)

    logger.info("Шумоподавление завершено.")
    return reduced_noise


def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Преобразует многоканальное аудио в моно путем усреднения всех каналов.

    Аргументы:
    audio_data (np.ndarray): Массив numpy с аудиоданными (несколько каналов).

    Возвращает:
    np.ndarray: Монофоническое аудио.
    """
    return np.mean(audio_data, axis=1)


def reduce_noise_segmented(audio_data: np.ndarray, sample_rate: int, chunk_size: int, prop_decrease: float,
                           stationary: bool, n_std_thresh_stationary: float) -> np.ndarray:
    """
    Удаляет шум из аудио, обрабатывая его сегментами с настройками подавления.

    Аргументы:
    audio_data (np.ndarray): Аудиоданные в виде массива numpy.
    sample_rate: Частота дискретизации.
    chunk_size: Размер сегмента для обработки.
    prop_decrease: Уровень подавления шума.
    stationary: Использовать ли стационарное шумоподавление.
    n_std_thresh_stationary: Порог стандартного отклонения для определения шума.

    Возвращает:
    np.ndarray: Очищенные аудиоданные.
    """
    num_chunks = len(audio_data) // chunk_size + 1
    denoised_audio = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio_data))

        # Извлечение сегмента
        chunk = audio_data[start_idx:end_idx]

        # Проверка длины сегмента и динамическое изменение nperseg, если сегмент слишком короткий
        nperseg = min(1024, len(chunk))

        # Применение шумоподавления на сегменте
        reduced_chunk = nr.reduce_noise(
            y=chunk, sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_fft=nperseg  # Динамическое изменение размера окна для STFT
        )

        # Добавление очищенного сегмента в общий список
        denoised_audio.extend(reduced_chunk)

    return np.array(denoised_audio)


# Пример использования:
if __name__ == "__main__":
    import soundfile as sf

    input_file_path = "../data/normalized/example.wav"

    # Загрузка аудиофайла
    data, sample_rate = sf.read(input_file_path)

    # Применение шумоподавления
    denoised_audio = denoise_audio_data(data, sample_rate, prop_decrease=0.7, stationary=True,
                                        n_std_thresh_stationary=1.2)

    # Пример сохранения результата, если нужно
    output_file_path = "../data/denoised/example.wav"
    sf.write(output_file_path, denoised_audio, sample_rate)
    logger.info(f"Очищенные аудиоданные сохранены в {output_file_path}.")
