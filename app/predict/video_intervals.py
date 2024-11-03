import csv
from pydub import AudioSegment


def load_intervals(file_path):
    """
    Загружает интервалы из CSV файла.
    :param file_path: Путь к файлу с интервалами.
    :return: Список интервалов в формате [(start, end), ...].
    """
    intervals = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Пропускаем заголовок
        for row in reader:
            start, end = map(float, row)
            intervals.append((start * 1000, end * 1000))  # Конвертируем в миллисекунды
    return intervals


def process_audio_channels(audio_path, intervals, output_path=None):
    """
    Разделяет звук между каналами в зависимости от таймкодов.
    :param audio_path: Путь к исходному аудио файлу.
    :param intervals: Список интервалов [(start, end), ...].
    :param output_path: Путь для сохранения обработанного аудио. Если None, сохраняется рядом с исходным.
    """
    audio = AudioSegment.from_file(audio_path)
    silent_segment = AudioSegment.silent(duration=10)  # Используется для контроля длины, если пустой канал

    # Разделение на левый и правый каналы
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]

    # Формирование правого и левого каналов с учетом интервалов
    processed_left = AudioSegment.silent(duration=len(audio))
    processed_right = AudioSegment.silent(duration=len(audio))

    current_pos = 0
    for start, end in intervals:
        # Левый канал вне интервала
        processed_left = processed_left.overlay(left_channel[current_pos:start], position=current_pos)
        # Правый канал внутри интервала
        processed_right = processed_right.overlay(right_channel[start:end], position=start)
        current_pos = end

    # Добавление оставшейся части вне интервалов
    if current_pos < len(audio):
        processed_left = processed_left.overlay(left_channel[current_pos:], position=current_pos)

    # Объединение каналов
    output_audio = AudioSegment.from_mono_audiosegments(processed_left, processed_right)

    # Сохранение выходного файла
    if output_path is None:
        output_path = audio_path.replace(".wav", "_processed.wav")
    output_audio.export(output_path, format="wav")
    print(f"Обработанный файл сохранен по пути: {output_path}")


# Пример использования
if __name__ == "__main__":
    audio_path = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #14 ХАКЕР ГРАБИТЕЛЬ • Майнкрафт.wav"  # Замените на путь к вашему файлу
    intervals_file = "processed_intervals.csv"  # Путь к файлу с интервалами
    output_path = "example_processed.wav"  # Путь для сохранения обработанного файла

    intervals = load_intervals(intervals_file)
    process_audio_channels(audio_path, intervals, output_path)
