from pydub import AudioSegment


def normalize_audio(input_file: str, output_file: str, target_dBFS: float = -20.0):
    """
    Нормализует громкость одного аудиофайла до заданного уровня dBFS.

    Аргументы:
    input_file (str): Путь к входному аудиофайлу.
    output_file (str): Путь для сохранения нормализованного аудиофайла.
    target_dBFS (float): Целевой уровень громкости в децибелах (по умолчанию -20 dBFS).
    """
    print(f"Нормализация файла: {input_file}")

    # Загружаем аудиофайл
    audio = AudioSegment.from_file(input_file)

    # Вычисляем разницу с целевым уровнем громкости
    change_in_dBFS = target_dBFS - audio.dBFS

    # Применяем изменение громкости
    normalized_audio = audio.apply_gain(change_in_dBFS)

    # Сохраняем нормализованный файл
    normalized_audio.export(output_file, format="wav")
    print(f"Файл {input_file} сохранен как {output_file} с нормализованной громкостью.")


# Пример использования:
if __name__ == "__main__":
    input_file_path = "data/nosilent/example.wav"
    output_file_path = "data/normalized/example.wav"

    normalize_audio(input_file_path, output_file_path)
