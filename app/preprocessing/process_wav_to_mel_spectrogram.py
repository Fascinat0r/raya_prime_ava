import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

from preprocessing.spectral_analysis import plot_mel_spectrogram


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
    # Преобразуем в моно, если аудио имеет несколько каналов
    if audio_segment.shape[0] > 1:
        audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)

    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return mel_spectrogram_transform(audio_segment)


def segment_audio_data(audio_data: torch.Tensor, segment_length_samples: int):
    """
    Разделяет аудиоданные на сегменты заданной длины.

    Аргументы:
    audio_data (Tensor): Аудиоданные в формате Tensor.
    segment_length_samples (int): Длина сегмента в сэмплах.

    Возвращает:
    Генератор, который возвращает сегменты аудиоданных.
    """
    num_samples = audio_data.shape[1]
    for start in range(0, num_samples, segment_length_samples):
        yield audio_data[:, start:start + segment_length_samples]


def divide_mel_spectrogram(mel_spectrogram: torch.Tensor, target_shape: tuple):
    """
    Разделяет мел-спектрограмму на более мелкие сегменты заданной формы.

    Аргументы:
    mel_spectrogram (Tensor): Исходная мел-спектрограмма.
    target_shape (tuple): Целевая форма сегмента (n_mels, количество временных фреймов).

    Возвращает:
    List[Tensor]: Список сегментов мел-спектрограммы.
    """
    target_mels, target_frames = target_shape
    mel_segments = []

    num_frames = mel_spectrogram.shape[2]  # Текущая длина по временной оси
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
                       n_mels: int):
    """
    Обрабатывает аудиофайл, преобразуя его в мел-спектрограммы и деля их на сегменты.

    Аргументы:
    filepath (str): Путь к аудиофайлу.
    segment_length_seconds (int): Длина сегмента в секундах.
    target_segment_shape (tuple): Целевая форма сегмента мел-спектрограммы.
    n_fft (int): Размер окна FFT.
    hop_length (int): Шаг окна.
    n_mels (int): Количество фильтров Мела.

    Возвращает:
    List[Tensor]: Список мел-спектрограмм для каждого сегмента.
    """
    # Загрузим аудиофайл
    waveform, sample_rate = torchaudio.load(filepath)

    # Преобразуем в моно, если есть несколько каналов
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Рассчитываем количество сэмплов в одном сегменте (например, 10 секунд)
    segment_length_samples = segment_length_seconds * sample_rate

    mel_segments = []
    leftover_mel = None

    # Обрабатываем каждый сегмент аудио по одному
    for audio_segment in segment_audio_data(waveform, segment_length_samples):
        mel_spectrogram = process_audio_segment_to_mel_spectrogram(audio_segment, sample_rate, n_fft, hop_length,
                                                                   n_mels)
        # Преобразуем мел-спектрограмму в децибелы
        mel_spectrogram = 10 * torch.log10(mel_spectrogram + 1e-10)

        if leftover_mel is not None:
            # Объединяем остаток с текущей мел-спектрограммой
            mel_spectrogram = torch.cat((leftover_mel, mel_spectrogram), dim=2)

        # Делим мел-спектрограмму на сегменты
        new_segments = divide_mel_spectrogram(mel_spectrogram, target_segment_shape)
        mel_segments.extend(new_segments)

        # Сохраняем остаток, если не хватило фреймов до полного сегмента
        num_frames = mel_spectrogram.shape[2]
        leftover_frames = num_frames % target_segment_shape[1]
        if leftover_frames > 0:
            leftover_mel = mel_spectrogram[:, :, -leftover_frames:]
        else:
            leftover_mel = None

    return mel_segments


if __name__ == "__main__":
    # Пример использования с передачей параметров напрямую
    filepath = "../data/raw/example.wav"
    segment_length_seconds = 10  # Длина сегмента в секундах
    target_segment_shape = (128, 128)  # Форма сегмента мел-спектрограммы
    n_fft = 2048  # Размер окна FFT
    hop_length = 256  # Шаг окна
    n_mels = 128  # Количество фильтров Мела

    # Преобразуем аудио в мел-спектрограммы сегментов
    mel_segments = process_audio_file(filepath, segment_length_seconds, target_segment_shape, n_fft, hop_length, n_mels)

    # Вывод информации
    print(f"Извлечено {len(mel_segments)} сегментов.")
    if mel_segments:
        print(f"Размер первого сегмента: {mel_segments[10].shape}")
        plot_mel_spectrogram(mel_segments[10], title="Пример мел-спектрограммы")
