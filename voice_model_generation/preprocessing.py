import librosa
import numpy as np
from utils.data_utils import augment_audio

from utils.my_logger import logger


# Генерация фрагментов аудио с фильтрацией тишины и шумов
def extract_audio_segments(file_list, sr=16000, segment_length=2, silence_threshold=20):
    """
    Разделяет аудиофайлы на сегменты фиксированной длины и фильтрует участки с тишиной.

    Args:
        file_list (list): Список аудиофайлов.
        sr (int): Частота дискретизации.
        segment_length (int): Длина каждого сегмента в секундах.
        silence_threshold (int): Минимальный уровень громкости (в дБ), ниже которого сегмент считается тишиной.

    Returns:
        list: Список аудиосегментов.
    """
    logger.info(
        f"Извлечение аудиосегментов из {len(file_list)} файлов с частотой дискретизации {sr} и длиной сегмента {segment_length} сек.")
    step = int(segment_length * sr)
    segments = []

    for file in file_list:
        logger.debug(f"Загрузка файла: {file}")
        audio, _ = librosa.load(file, sr=sr)
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)

        # Удаление участков тишины
        non_silent_intervals = librosa.effects.split(audio, top_db=silence_threshold)
        filtered_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])

        # Разделение на сегменты с перекрытием на 50%
        for start in range(0, len(filtered_audio) - step, step // 2):
            segment = filtered_audio[start:start + step]
            if len(segment) == step:
                segments.append(segment)

        logger.debug(f"Извлечено {len(segments)} сегментов из файла {file} после фильтрации тишины")

    return segments


# Нормализация количества сегментов с использованием аугментации
def normalize_segments(raya_segments, other_segments, noise_segments):
    """
    Нормализует количество сегментов с помощью аугментации данных.

    Args:
        raya_segments (list): Сегменты для голоса Райи.
        other_segments (list): Сегменты для других голосов.
        noise_segments (list): Сегменты для шумов.

    Returns:
        tuple: Нормализованные списки сегментов.
    """
    logger.info("Нормализация количества сегментов...")
    max_len = max(len(raya_segments), len(other_segments), len(noise_segments))
    logger.info(f"Максимальное количество сегментов: {max_len}")

    # Аугментация данных вместо дублирования
    raya_segments = augment_audio(raya_segments, target_length=max_len)
    other_segments = augment_audio(other_segments, target_length=max_len)
    noise_segments = augment_audio(noise_segments, target_length=max_len)

    logger.info(
        f"После нормализации: Голос Райи: {len(raya_segments)}, Другие голоса: {len(other_segments)}, Шумы: {len(noise_segments)}")
    return raya_segments, other_segments, noise_segments
