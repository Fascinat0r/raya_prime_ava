import os

import soundfile as sf


def segment_audio(input_file: str, output_folder: str, segment_length: float = 5.0, overlap: float = 0.5):
    """
    Разделяет аудиофайл на сегменты заданной длины и перекрытием.

    Аргументы:
    input_file (str): Путь к входному .wav файлу.
    output_folder (str): Путь для сохранения сегментов.
    segment_length (float): Длина каждого сегмента в секундах (по умолчанию 5.0 секунд).
    overlap (float): Степень перекрытия между сегментами (значения от 0 до 1, по умолчанию 0.5).
    """
    # Убеждаемся, что выходная папка существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Загружаем аудиофайл
    data, sample_rate = sf.read(input_file)

    # Рассчитываем длину каждого сегмента и шаг между сегментами в выборках
    segment_samples = int(segment_length * sample_rate)
    step_size = int(segment_samples * (1 - overlap))

    # Количество сегментов, которые можно получить
    total_segments = (len(data) - segment_samples) // step_size + 1

    print(
        f"Разделение файла на {total_segments} сегментов, длина: {segment_length} секунд, перекрытие: {overlap * 100:.0f}%")

    # Перебираем и создаем сегменты
    for i in range(total_segments):
        start_sample = i * step_size
        end_sample = start_sample + segment_samples

        # Извлекаем сегмент
        segment = data[start_sample:end_sample]

        # Генерируем имя выходного файла
        output_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}_segment_{i + 1}.wav"
        output_path = os.path.join(output_folder, output_filename)

        # Сохранение сегмента
        sf.write(output_path, segment, sample_rate)
        print(f"Сегмент {i + 1} сохранен: {output_path}")


# Пример использования:
if __name__ == "__main__":
    input_file_path = "data/normalized/example.wav"
    output_folder_path = "data/segments"

    # Разделение на сегменты длиной 5 секунд с перекрытием 50%
    segment_audio(input_file_path, output_folder_path, segment_length=5.0, overlap=0.5)
