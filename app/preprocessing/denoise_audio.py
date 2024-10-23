import noisereduce as nr
import numpy as np

from app.utils.logger import get_logger

logger = get_logger("denoise_audio")


def denoise_audio_data(audio_data, sample_rate, chunk_size=50000, prop_decrease=0.8,
                       stationary=False, n_std_thresh_stationary=1.0, freq_mask_smooth_hz=700) -> np.ndarray:
    """
    Удаляет шум из аудиоданных, обрабатывая их сегментами и преобразуя многоканальные файлы в моно.

    Аргументы:
    audio_data (numpy.ndarray): Аудиоданные.
    sample_rate (int): Частота дискретизации.
    chunk_size (int): Размер сегмента для обработки (по умолчанию 50000 выборок).
    prop_decrease (float): Уровень подавления шума (по умолчанию 0.8, максимальное значение 1.0).
    stationary (bool): Использовать ли стационарный шумоподавитель (False для спектрального подавления).
    n_std_thresh_stationary (float): Порог стандартного отклонения для определения шума.
    freq_mask_smooth_hz (float): Частота сглаживания частотной маски.
    """
    logger.info("Запуск шумоподавления аудиоданных...")

    # Проверка количества каналов
    if len(audio_data.shape) > 1:
        logger.error("Многоканальные аудиоданныe")
        raise ValueError("Многоканальные аудиоданныe не поддерживаются. Преобразуйте их в моно.")

    # Применение шумоподавления сегментами
    logger.info("Обработка аудиоданных и удаление шума...")
    reduced_noise = reduce_noise_segmented(audio_data, sample_rate, chunk_size, prop_decrease,
                                           stationary, n_std_thresh_stationary, freq_mask_smooth_hz)

    return reduced_noise


def reduce_noise_segmented(audio_data, sample_rate, chunk_size, prop_decrease, stationary, n_std_thresh_stationary,
                           freq_mask_smooth_hz):
    """
    Удаляет шум из аудио, обрабатывая его сегментами с настройками подавления.

    Аргументы:
    audio_data: Аудиоданные в виде массива numpy.
    sample_rate: Частота дискретизации.
    chunk_size: Размер сегмента для обработки.
    prop_decrease: Уровень подавления шума.
    stationary: Использовать ли стационарное шумоподавление.
    n_std_thresh_stationary: Порог стандартного отклонения для определения шума.
    freq_mask_smooth_hz (float): Частота сглаживания частотной маски.

    Возвращает:
    Очищенные аудиоданные.
    """
    num_chunks = len(audio_data) // chunk_size + 1
    denoised_audio = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio_data))

        # Извлечение сегмента
        chunk = audio_data[start_idx:end_idx]

        # Применение шумоподавления на сегменте
        reduced_chunk = nr.reduce_noise(
            y=chunk,
            sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            freq_mask_smooth_hz=freq_mask_smooth_hz
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
