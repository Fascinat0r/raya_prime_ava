import os

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

from app.utils.logger import get_logger

logger = get_logger("utils")


def audiosegment_to_numpy(audio_segment: AudioSegment) -> np.ndarray:
    """
    Преобразует объект AudioSegment в массив numpy.
    """
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
    return samples.astype(np.float32) / (2 ** 15)


def load_wav_file(file_path):
    """
    Загружает аудиофайл WAV формата с использованием torchaudio и выводит его основные характеристики.

    Аргументы:
    file_path (str): Путь к WAV файлу.

    Возвращает:
    waveform (Tensor): Аудиосигнал в формате Tensor.
    sample_rate (int): Частота дискретизации.
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return None, None

    waveform, sample_rate = torchaudio.load(file_path)

    # Преобразуем стерео в моно (если требуется)
    if waveform.shape[0] > 1:
        logger.info(f"Преобразование стерео в моно.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    logger.info(f"Загружен файл: {file_path}")
    logger.info(f"Частота дискретизации: {sample_rate} Гц")
    logger.info(f"Количество сэмплов: {waveform.shape[1]}")
    logger.info(f"Количество каналов: {waveform.shape[0]}")

    return waveform, sample_rate


def tensor_to_audiosegment(tensor: torch.Tensor, sample_rate: int, num_channels: int = 1, sample_width: int = 2):
    """
    Преобразует Tensor аудиосигнала в объект AudioSegment.

    Аргументы:
    tensor (Tensor): Аудиоданные в формате Tensor.
    sample_rate (int): Частота дискретизации.
    num_channels (int): Количество каналов (1 - моно, 2 - стерео).
    sample_width (int): Ширина сэмпла в байтах (2 байта = 16 бит).

    Возвращает:
    AudioSegment: Аудиофайл в формате AudioSegment.
    """
    # Преобразуем Tensor в numpy
    numpy_data = tensor.cpu().numpy()

    # Убедимся, что данные имеют правильную форму (один канал)
    if num_channels == 1:
        numpy_data = np.squeeze(numpy_data)

    # Преобразуем в формат int16 (если это 16-битное аудио)
    if sample_width == 2:
        numpy_data = (numpy_data * 32767).astype(np.int16)

    # Преобразуем numpy в байты
    audio_data = numpy_data.tobytes()

    # Создаем AudioSegment из сырых байтовых данных
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=sample_width,  # 2 байта = 16 бит
        frame_rate=sample_rate,
        channels=num_channels
    )

    return audio_segment
