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
    """
    logger.debug(f"Processing audio segment: shape={audio_segment.shape}, sample_rate={sample_rate}, "
                f"n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")

    if audio_segment.shape[1] < n_fft:
        logger.warning(f"Audio segment too short for n_fft={n_fft}. Segment length: {audio_segment.shape[1]}")
        return None

    if audio_segment.shape[0] > 1:
        audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)
        logger.debug(f"Converted to mono: shape={audio_segment.shape}")

    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_spectrogram_transform(audio_segment)
    logger.debug(f"Generated mel spectrogram: shape={mel_spectrogram.shape}")
    return mel_spectrogram


def segment_audio_data(audio_data: torch.Tensor, segment_length_samples: int):
    """
    Разделяет аудиоданные на сегменты фиксированной длины (без перекрытия).
    """
    num_samples = audio_data.shape[1]
    logger.debug(f"Segmenting audio data: total_samples={num_samples}, segment_length={segment_length_samples}")

    for start in range(0, num_samples, segment_length_samples):
        end = min(start + segment_length_samples, num_samples)
        segment = audio_data[:, start:end]
        logger.debug(f"Yielding audio segment: start_frame={start}, end_frame={end}, segment_shape={segment.shape}")
        yield segment, start


def divide_mel_spectrogram(mel_spectrogram: torch.Tensor, target_shape: tuple, overlap: float, start_frame: int, hop_length: int):
    """
    Делит мел-спектрограмму на части заданного размера с перекрытием, возвращая стартовые фреймы сегментов.
    """
    target_mels, target_frames = target_shape
    mel_segments = []
    frame_indices = []
    step_size = int(target_frames * (1 - overlap))
    num_frames = mel_spectrogram.shape[2]

    logger.debug(f"Dividing mel spectrogram: shape={mel_spectrogram.shape}, target_shape={target_shape}, "
                f"step_size={step_size}, overlap={overlap}")

    for start in range(0, num_frames - target_frames + 1, step_size):
        segment = mel_spectrogram[:, :, start:start + target_frames]
        # Корректируем `start_frame` в аудиофреймах по hop_length
        frame_in_audio_samples = start_frame + start * hop_length
        logger.debug(f"Extracted segment: start_frame_in_audio={frame_in_audio_samples}, end_frame_in_audio={frame_in_audio_samples + target_frames * hop_length}, "
                    f"segment_shape={segment.shape}")

        if segment.shape[2] == target_frames:
            mel_segments.append(segment)
            frame_indices.append(frame_in_audio_samples)

    logger.debug(f"Total segments created: {len(mel_segments)}")
    return mel_segments, frame_indices


def process_audio_file(filepath: str,
                       segment_length_seconds: int,
                       target_segment_shape: tuple,
                       n_fft: int,
                       hop_length: int,
                       n_mels: int,
                       overlap: float):
    """
    Обрабатывает аудиофайл, преобразуя его в мел-спектрограммы и деля на целевые сегменты.
    """
    logger.debug(f"Processing audio file: {filepath}")
    waveform, sample_rate = torchaudio.load(filepath)
    logger.debug(f"Loaded waveform: shape={waveform.shape}, sample_rate={sample_rate}")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        logger.debug(f"Converted to mono: shape={waveform.shape}")

    # Длина 10-секундного блока в сэмплах
    segment_length_samples = int(segment_length_seconds * sample_rate)
    logger.debug(f"10-second segment length in samples: {segment_length_samples}")

    mel_segments = []
    segment_metadata = []

    # Шаг 1: Разбиваем аудио на 10-секундные блоки
    for audio_segment, start_frame in segment_audio_data(waveform, segment_length_samples):
        # Шаг 2: Преобразуем блок в мел-спектрограмму
        mel_spectrogram = process_audio_segment_to_mel_spectrogram(audio_segment, sample_rate, n_fft, hop_length,
                                                                   n_mels)
        if mel_spectrogram is None:
            logger.warning(f"Skipping segment starting at frame {start_frame} due to short length.")
            continue

        # Преобразуем спектрограмму в логарифмическую шкалу
        mel_spectrogram = 10 * torch.log10(mel_spectrogram + 1e-10)

        # Шаг 3: Разделяем мел-спектрограмму на целевые сегменты (64, 64) с перекрытием
        new_segments, frame_indices = divide_mel_spectrogram(mel_spectrogram, target_segment_shape, overlap, start_frame, hop_length)
        mel_segments.extend(new_segments)

        # Записываем стартовые фреймы в аудиофреймах
        segment_metadata.extend([(frame_idx, segment) for frame_idx, segment in zip(frame_indices, new_segments)])

    # Преобразуем стартовые фреймы в секунды
    segment_metadata = [(frame_idx / float(sample_rate), segment) for frame_idx, segment in segment_metadata]

    logger.debug(f"Processed {len(mel_segments)} mel spectrogram segments from file: {filepath}")
    return segment_metadata


if __name__ == "__main__":
    filepath = "../data/raw/example.wav"
    segment_length_seconds = 10
    target_segment_shape = (64, 64)
    n_fft = 2048
    hop_length = 256
    n_mels = 64
    overlap = 0.5

    logger.debug("Starting audio processing")
    mel_segments_with_times = process_audio_file(filepath, segment_length_seconds, target_segment_shape, n_fft,
                                                 hop_length, n_mels, overlap)

    logger.debug(f"Extracted {len(mel_segments_with_times)} segments with start times.")
    for start_time, segment in mel_segments_with_times:
        logger.debug(f"Segment start time: {start_time:.2f} seconds, segment shape: {segment.shape}")
