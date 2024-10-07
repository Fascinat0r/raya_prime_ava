import os

import librosa

from utils.my_logger import logger


def load_all_files(directory, target_sr=16000):
    """
    Загружает все аудиофайлы из указанной директории, приводя их к единой частоте дискретизации.

    Args:
        directory (str): Путь к директории с аудиофайлами.
        target_sr (int): Целевая частота дискретизации для всех файлов.

    Returns:
        list: Список загруженных аудиофайлов и их имен.
    """
    logger.info(f"Загрузка аудиофайлов из директории: {directory}")
    files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".wav")]
    loaded_files = []
    filenames = []

    for file in files:
        try:
            audio, sr = librosa.load(file, sr=None)  # Загружаем файл с его исходной частотой дискретизации
            if sr != target_sr:
                logger.info(f"Преобразование частоты дискретизации {sr} -> {target_sr} для файла: {file}")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)  # Приведение частоты к target_sr
            loaded_files.append(audio)
            filenames.append(os.path.basename(file))
        except Exception as e:
            logger.warning(f"Ошибка загрузки файла {file}: {e}")

    logger.info(f"Загружено {len(loaded_files)} файлов в {directory} с частотой дискретизации {target_sr}")
    return loaded_files, filenames
