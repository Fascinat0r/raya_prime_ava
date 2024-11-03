import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

from logger import get_logger

logger = get_logger(__name__)


def process_audio_segment_to_mel_spectrogram(audio_segment: torch.Tensor,
                                             sample_rate: int,
                                             n_fft: int,
                                             hop_length: int,
                                             n_mels: int):
    """
    Преобразует сегмент аудио в мел-спектрограмму.

    Аргументы:
    audio_segment (Tensor): Сегмент аудиоданных.
    sample_rate (int): Частота дискретизации.
    n_fft (int): Размер окна FFT.
    hop_length (int): Шаг окна.
    n_mels (int): Количество фильтров Мела.

    Возвращает:
    Tensor: Мел-спектрограмма для данного сегмента.
    """
    if audio_segment.shape[1] < n_fft:
        logger.warning(
            f"Сегмент аудио слишком короткий для n_fft={n_fft}. Уменьшите n_fft или увеличьте длину сегмента.")
        return None

    if audio_segment.shape[0] > 1:
        audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)

    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return mel_spectrogram_transform(audio_segment)


def segment_audio_data(audio_data: torch.Tensor, segment_length_samples: int, overlap: float, sample_rate: int):
    """
    Разделяет аудиоданные на сегменты заданной длины с перекрытием.

    Аргументы:
    audio_data (Tensor): Аудиоданные.
    segment_length_samples (int): Длина сегмента в сэмплах.
    overlap (float): Процент перекрытия между сегментами.
    sample_rate (int): Частота дискретизации.

    Возвращает:
    Generator: Генератор сегментов и времени начала каждого сегмента.
    """
    step_size = int(segment_length_samples * (1 - overlap))
    for start in range(0, audio_data.shape[1], step_size):
        end = min(start + segment_length_samples, audio_data.shape[1])  # Последний сегмент может быть короче
        segment = audio_data[:, start:end]
        start_time = start / sample_rate  # Время начала сегмента в секундах
        yield segment, start_time


def divide_mel_spectrogram(mel_spectrogram: torch.Tensor, target_shape: tuple):
    """
    Делит мел-спектрограмму на части заданного размера.

    Аргументы:
    mel_spectrogram (Tensor): Мел-спектрограмма для деления.
    target_shape (tuple): Целевая форма (количество мел-фильтров, временные фреймы).

    Возвращает:
    List[Tensor]: Список сегментов мел-спектрограммы.
    """
    target_mels, target_frames = target_shape
    mel_segments = []

    num_frames = mel_spectrogram.shape[2]
    for start in range(0, num_frames, target_frames):
        segment = mel_spectrogram[:, :, start:start + target_frames]
        if segment.shape[2] == target_frames:
            mel_segments.append(segment)

    return mel_segments


def process_audio_file(filepath: str,
                       segment_length_seconds: int,
                       target_segment_shape: tuple,
                       n_fft: int,
                       hop_length: int,
                       n_mels: int,
                       overlap: float):
    """
    Обрабатывает аудиофайл, преобразуя его в мел-спектрограммы и деля на сегменты.

    Аргументы:
    filepath (str): Путь к аудиофайлу.
    segment_length_seconds (int): Длина сегмента в секундах.
    target_segment_shape (tuple): Целевая форма сегмента мел-спектрограммы.
    n_fft (int): Размер окна FFT.
    hop_length (int): Шаг окна.
    n_mels (int): Количество фильтров Мела.
    overlap (float): Процент перекрытия между сегментами.

    Возвращает:
    List[tuple]: Список кортежей, содержащих время начала и сегменты мел-спектрограммы.
    """
    logger.info(f"Обработка аудиофайла {filepath}")
    waveform, sample_rate = torchaudio.load(filepath)

    # Преобразование в моно, если есть несколько каналов
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    segment_length_samples = segment_length_seconds * sample_rate
    mel_segments = []
    segment_metadata = []
    leftover_mel = None

    for audio_segment, start_time in segment_audio_data(waveform, segment_length_samples, overlap, sample_rate):
        mel_spectrogram = process_audio_segment_to_mel_spectrogram(audio_segment, sample_rate, n_fft, hop_length,
                                                                   n_mels)
        if mel_spectrogram is None:
            continue
        mel_spectrogram = 10 * torch.log10(mel_spectrogram + 1e-10)

        if leftover_mel is not None:
            mel_spectrogram = torch.cat((leftover_mel, mel_spectrogram), dim=2)

        new_segments = divide_mel_spectrogram(mel_spectrogram, target_segment_shape)
        mel_segments.extend(new_segments)

        segment_metadata.extend([(start_time + i * (target_segment_shape[1] * hop_length / sample_rate), segment)
                                 for i, segment in enumerate(new_segments)])

        num_frames = mel_spectrogram.shape[2]
        leftover_frames = num_frames % target_segment_shape[1]
        leftover_mel = mel_spectrogram[:, :, -leftover_frames:] if leftover_frames > 0 else None

    return segment_metadata


if __name__ == "__main__":
    filepath = "../data/raw/example.wav"
    segment_length_seconds = 10
    target_segment_shape = (64, 64)
    n_fft = 2048
    hop_length = 256
    n_mels = 64
    overlap = 0.3

    mel_segments_with_times = process_audio_file(filepath, segment_length_seconds, target_segment_shape, n_fft,
                                                 hop_length, n_mels, overlap)

    print(f"Извлечено {len(mel_segments_with_times)} сегментов с таймкодами.")
    for start_time, segment in mel_segments_with_times:
        print(f"Start time: {start_time:.2f} seconds")
