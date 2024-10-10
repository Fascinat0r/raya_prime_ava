from pydub import AudioSegment, silence

from app.utils.logger import get_logger

logger = get_logger("filter_silent")

def remove_silence(input_wav: str, output_wav: str, silence_thresh: int = -40, min_silence_len: int = 500):
    """
    Удаляет тихие моменты без голоса из .wav файла.

    Аргументы:
    input_wav (str): Путь к входному .wav файлу.
    output_wav (str): Путь для сохранения выходного .wav файла.
    silence_thresh (int): Порог тишины в децибелах (значения от 0 до -100, чем меньше, тем тише).
    min_silence_len (int): Минимальная длина тишины в миллисекундах, которую нужно удалять.
    """
    # Загружаем аудиофайл
    audio = AudioSegment.from_wav(input_wav)

    # Разделяем аудио на сегменты, которые не содержат тишину
    chunks = silence.split_on_silence(audio,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh)

    # Объединяем сегменты и сохраняем результат
    if chunks:
        combined_audio = chunks[0]
        for chunk in chunks[1:]:
            combined_audio += chunk
        combined_audio.export(output_wav, format="wav")
        logger.info(f"Тишина удалена, результат сохранен в {output_wav}")
    else:
        logger.info("Все аудио распознано как тишина.")


# Пример использования
if __name__ == "__main__":
    input_file = "../data/raw/example.wav"
    output_file = "../data/nosilent/example.wav"
    remove_silence(input_file, output_file)
