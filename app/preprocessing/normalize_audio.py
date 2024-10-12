from pydub import AudioSegment

from app.utils.logger import get_logger

logger = get_logger("normalize_audio")


def normalize_audio_data(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """
    Нормализует громкость аудиоданных до заданного уровня dBFS.

    Аргументы:
    audio (AudioSegment): Загруженные аудиоданные.
    target_dBFS (float): Целевой уровень громкости в децибелах (по умолчанию -20 dBFS).

    Возвращает:
    AudioSegment: Нормализованные аудиоданные.
    """
    logger.info(f"Нормализация аудиоданных.")

    # Вычисляем разницу с целевым уровнем громкости
    change_in_dBFS = target_dBFS - audio.dBFS

    # Применяем изменение громкости
    normalized_audio = audio.apply_gain(change_in_dBFS)

    logger.info(f"Нормализация завершена. Изменение громкости: {change_in_dBFS} dB")
    return normalized_audio


# Пример использования:
if __name__ == "__main__":
    input_file_path = "../data/nosilent/example.wav"

    # Загрузка аудио
    audio = AudioSegment.from_file(input_file_path)

    # Нормализация
    normalized_audio = normalize_audio_data(audio)

    if normalized_audio:
        # Сохранение при необходимости
        output_file_path = "../data/normalized/example.wav"
        normalized_audio.export(output_file_path, format="wav")
        logger.info(f"Нормализованное аудио сохранено как {output_file_path}")
