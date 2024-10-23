from pydub import AudioSegment, silence

from logger import get_logger

logger = get_logger("filter_silent")


def remove_silence_from_data(audio: AudioSegment, silence_thresh: int = -40,
                             min_silence_len: int = 50) -> AudioSegment:
    """
    Удаляет тихие моменты без голоса из аудиоданных.

    Аргументы:
    audio (AudioSegment): Загруженные аудиоданные.
    silence_thresh (int): Порог тишины в децибелах (значения от 0 до -100, чем меньше, тем тише).
    min_silence_len (int): Минимальная длина тишины в миллисекундах, которую нужно удалять.

    Возвращает:
    AudioSegment: Аудио без тишины.
    """
    # Разделяем аудио на сегменты, которые не содержат тишину
    chunks = silence.split_on_silence(audio,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh)

    # Объединяем сегменты и возвращаем результат
    if chunks:
        combined_audio = chunks[0]
        for chunk in chunks[1:]:
            combined_audio += chunk
        logger.info("Тишина удалена.")
        return combined_audio
    else:
        logger.info("Все аудио распознано как тишина.")
        return None


# Пример использования
if __name__ == "__main__":
    input_file = "../data/raw/example.wav"
    audio = AudioSegment.from_wav(input_file)

    # Удаляем тишину из аудиоданных
    processed_audio = remove_silence_from_data(audio)

    if processed_audio:
        output_file = "../data/nosilent/example.wav"
        processed_audio.export(output_file, format="wav")
        logger.info(f"Результат сохранен в {output_file}")
