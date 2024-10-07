import os


def get_all_audio_files(folder_path, file_extension=".wav"):
    """Возвращает список всех аудиофайлов с заданным расширением в папке."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_extension)]


def timecode_format(seconds):
    """Преобразует время в секундах в формат 'чч:мм:сс'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
