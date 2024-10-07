from utils.data_utils import augment_audio
from utils.my_logger import logger


# Генерация фрагментов аудио с фильтрацией тишины и шумов
def extract_audio_segments(audio_data_list, sr=16000, segment_length=2):
    """
    Разделяет аудиомассивы на сегменты фиксированной длины.

    Args:
        audio_data_list (list): Список загруженных аудиомассивов.
        sr (int): Частота дискретизации.
        segment_length (int): Длина каждого сегмента в секундах.

    Returns:
        list: Список аудиосегментов.
    """
    logger.info(
        f"Извлечение аудиосегментов из {len(audio_data_list)} загруженных массивов с частотой {sr} и длиной сегмента {segment_length} сек.")
    step = int(segment_length * sr)
    segments = []

    for audio in audio_data_list:
        logger.debug(f"Обработка загруженного аудиофрагмента длиной {len(audio)}.")
        for start in range(0, len(audio) - step, step // 2):  # Перекрытие на 50%
            segment = audio[start:start + step]
            if len(segment) == step:
                segments.append(segment)

    logger.info(f"Общее количество извлеченных сегментов: {len(segments)}")
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
