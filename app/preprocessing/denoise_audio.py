import noisereduce as nr
import numpy as np
import soundfile as sf


def denoise_wav(input_file: str, output_file: str, chunk_size: int = 50000, prop_decrease: float = 0.8,
                stationary: bool = False, n_std_thresh_stationary: float = 1.0):
    """
    Удаляет шум из .wav файла, обрабатывая его сегментами и преобразуя многоканальные файлы в моно.

    Аргументы:
    input_file (str): Путь к входному .wav файлу.
    output_file (str): Путь для сохранения выходного .wav файла.
    chunk_size (int): Размер сегмента для обработки (по умолчанию 50000 выборок).
    prop_decrease (float): Уровень подавления шума (по умолчанию 0.8, максимальное значение 1.0).
    stationary (bool): Использовать ли стационарный шумоподавитель (False для спектрального подавления).
    n_std_thresh_stationary (float): Порог стандартного отклонения для определения шума.
    """
    print(f"Удаление шума из файла: {input_file}")

    # Загрузка аудиофайла
    data, sample_rate = sf.read(input_file)

    # Проверка количества каналов и преобразование в моно
    if len(data.shape) > 1:
        print("Аудиофайл многоканальный, преобразование в моно.")
        data = convert_to_mono(data)

    # Применение шумоподавления сегментами
    print("Обработка аудио и удаление шума...")
    reduced_noise = reduce_noise_segmented(data, sample_rate, chunk_size, prop_decrease, stationary,
                                           n_std_thresh_stationary)

    # Сохранение очищенного аудио
    sf.write(output_file, reduced_noise, sample_rate)
    print(f"Шум удален, файл сохранен в {output_file}")


def convert_to_mono(audio_data):
    """
    Преобразует многоканальное аудио в моно путем усреднения всех каналов.

    Аргументы:
    audio_data: Массив numpy с аудиоданными (несколько каналов).

    Возвращает:
    Монофоническое аудио.
    """
    return np.mean(audio_data, axis=1)


def reduce_noise_segmented(audio_data, sample_rate, chunk_size, prop_decrease, stationary, n_std_thresh_stationary):
    """
    Удаляет шум из аудио, обрабатывая его сегментами с настройками подавления.

    Аргументы:
    audio_data: Аудиоданные в виде массива numpy.
    sample_rate: Частота дискретизации.
    chunk_size: Размер сегмента для обработки.
    prop_decrease: Уровень подавления шума.
    stationary: Использовать ли стационарное шумоподавление.
    n_std_thresh_stationary: Порог стандартного отклонения для определения шума.

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
            y=chunk, sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=stationary,
            n_std_thresh_stationary=n_std_thresh_stationary
        )

        # Добавление очищенного сегмента в общий список
        denoised_audio.extend(reduced_chunk)

    return np.array(denoised_audio)


# Пример использования:
if __name__ == "__main__":
    input_file_path = "../data/normalized/example.wav"
    output_file_path = "../data/denoised/example.wav"

    # Установить параметры для более мягкого шумоподавления
    denoise_wav(input_file_path, output_file_path, prop_decrease=0.7, stationary=True, n_std_thresh_stationary=1.2)
