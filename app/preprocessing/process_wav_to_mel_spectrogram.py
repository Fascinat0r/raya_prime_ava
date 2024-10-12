import logging

import torch
import torchaudio
import torchaudio.functional as F

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mel_fbank_conversion")


def process_audio_to_mel_spectrogram(audio_data: torch.Tensor,
                                     sample_rate: int,
                                     expected_shape=(64, 64),
                                     overlap=0.3,
                                     hop_length=512,
                                     n_fft=2048,
                                     n_mels=64):
    """
    Обрабатывает аудиоданные с сегментацией и конвертацией в мел-спектрограммы.

    Аргументы:
    audio_data (Tensor): Аудиоданные в формате Tensor.
    sample_rate (int): Частота дискретизации.
    expected_shape (tuple): Ожидаемый размер мел-спектрограммы (строки, столбцы).
    overlap (float): Процент перекрытия сегментов.
    hop_length (int): Шаг окна.
    n_fft (int): Размер окна FFT.
    n_mels (int): Количество фильтров Мела.

    Возвращает:
    List[Tensor]: Список мел-спектрограмм размером expected_shape.
    """
    expected_rows, expected_columns = expected_shape

    # Рассчитать длину сегмента в сэмплах для нужного количества фреймов
    segment_length_samples = hop_length * (expected_columns - 1) + n_fft
    logger.info(f"Количество сэмплов в сегменте: {segment_length_samples}")

    # Рассчитать перекрытие в сэмплах
    overlap_samples = int(overlap * segment_length_samples)

    # Разделение на сегменты
    segments = segment_waveform(audio_data, segment_length_samples, overlap_samples)

    mel_spectrogram_list = []
    mel_filter_banks = calculate_mel_fbanks(sample_rate, n_mels=n_mels, n_fft=n_fft)

    for i, segment in enumerate(segments):
        # Рассчитываем STFT для каждого сегмента
        window = torch.hann_window(n_fft).to(segment.device)
        stft = torch.stft(segment, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        magnitude = torch.abs(stft) ** 2

        # Применение мел-фильтров к спектрограмме (создание мел-спектрограммы)
        mel_spectrogram = torch.matmul(magnitude.permute(0, 2, 1), mel_filter_banks)

        # Обрезаем или дополняем до expected_shape
        if mel_spectrogram.shape[1] >= expected_rows:
            logger.debug(f"Размер сегмента {i + 1} больше {expected_rows}. {mel_spectrogram.shape[1]}")
            mel_spectrogram = mel_spectrogram[:, :expected_rows, :]
        else:
            logger.debug(f"Размер сегмента {i + 1} меньше {expected_rows}. {mel_spectrogram.shape[1]}")
            padding = torch.zeros(
                (mel_spectrogram.shape[0], expected_rows - mel_spectrogram.shape[1], mel_spectrogram.shape[2])).to(
                mel_spectrogram.device)
            mel_spectrogram = torch.cat((mel_spectrogram, padding), dim=1)

        # Проверка на правильность формы
        if mel_spectrogram.shape == (1, expected_rows, expected_columns):
            mel_spectrogram_list.append(mel_spectrogram)
            logger.debug(f"Сегмент {i + 1} обработан. Форма мел-спектрограммы: {mel_spectrogram.shape}")
        else:
            logger.info(f"Сегмент {i + 1} пропущен. Неправильная форма: {mel_spectrogram.shape}")

    logger.info(f"Всего извлечено сегментов: {len(mel_spectrogram_list)}")
    return mel_spectrogram_list


def segment_waveform(waveform, segment_length_samples, overlap_samples):
    """
    Разделяет аудиоданные на сегменты с заданной длиной и перекрытием.

    Аргументы:
    waveform (Tensor): Аудиоданные.
    segment_length_samples (int): Длина сегмента в сэмплах.
    overlap_samples (int): Количество сэмплов для перекрытия.

    Возвращает:
    List[Tensor]: Список сегментов аудиоданных.
    """
    segments = []
    step_size = segment_length_samples - overlap_samples
    for start in range(0, waveform.shape[1] - segment_length_samples + 1, step_size):
        segment = waveform[:, start:start + segment_length_samples]
        segments.append(segment)

    logger.info(f"Разделено на {len(segments)} сегментов.")
    return segments


def calculate_mel_fbanks(sample_rate, n_mels=64, f_min=0.0, f_max=None, n_fft=2048):
    """
    Рассчитывает мел-фильтры (Mel FBanks).

    Аргументы:
    sample_rate (int): Частота дискретизации.
    n_mels (int): Количество фильтров Мела.
    f_min (float): Минимальная частота.
    f_max (float): Максимальная частота. Если None, будет равно половине частоты дискретизации.
    n_fft (int): Размер окна FFT.

    Возвращает:
    Tensor: Мел-фильтры.
    """
    if f_max is None:
        f_max = sample_rate / 2

    mel_fbanks = F.melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=f_min, f_max=f_max, n_mels=n_mels,
                                   sample_rate=sample_rate)
    logger.info(f"Рассчитаны мел-фильтры. Форма: {mel_fbanks.shape}")
    return mel_fbanks


# Пример использования:
if __name__ == "__main__":
    # Для демонстрации можно загрузить данные напрямую из файла, но в реальности вы можете использовать аудиоданные в виде тензора
    waveform, sample_rate = torchaudio.load("../data/normalized/example.wav")

    # Обработка аудиоданных и извлечение мел-спектрограмм с ожидаемой формой 64x64
    mel_spectrograms = process_audio_to_mel_spectrogram(waveform, sample_rate, expected_shape=(64, 64), hop_length=512,
                                                        n_fft=2048)

    # Проверка результата
    if mel_spectrograms:
        logger.info(f"Извлечено {len(mel_spectrograms)} мел-спектрограмм.")
        logger.info(f"Размер первого сегмента: {mel_spectrograms[0].shape}")
