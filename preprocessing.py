import logging
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, stft, istft

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Директория для сохранения промежуточных файлов
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Функция загрузки аудиофайла с помощью SciPy
def load_audio(file_path):
    logging.info(f"Загрузка аудиофайла: {file_path}")
    sr, y = wavfile.read(file_path)
    if y.ndim > 1:  # Преобразование в моно, если аудио стерео
        y = np.mean(y, axis=1)
    y = y.astype(np.float32) / 32768.0  # Нормализация значений до [-1, 1]
    logging.info(f"Файл загружен. Длина сигнала: {len(y)} samples, Частота дискретизации: {sr}")
    return y, sr


# Нормализация сигнала
def normalize_audio(y):
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y


# Преэмфазис для усиления высоких частот
def apply_preemphasis(y, coeff=0.97):
    return np.append(y[0], y[1:] - coeff * y[:-1])


# Низкочастотный фильтр для удаления помех
def lowpass_filter(y, sr, cutoff=100, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered


# Разделение гармонических и перкуссионных элементов с помощью STFT и iSTFT
def hpss_scipy(y, sr, kernel_size=512):
    f, t, Zxx = stft(y, sr, nperseg=kernel_size)
    # Разделение по порогу на гармоническую и перкуссионную составляющие
    magnitude, phase = np.abs(Zxx), np.angle(Zxx)
    harmonic_mask = magnitude > np.median(magnitude)
    percussive_mask = ~harmonic_mask

    # Применение масок для извлечения составляющих
    Zxx_harmonic = Zxx * harmonic_mask
    Zxx_percussive = Zxx * percussive_mask

    # Преобразование обратно в временной ряд
    _, y_harmonic = istft(Zxx_harmonic, sr)
    _, y_percussive = istft(Zxx_percussive, sr)

    return y_harmonic, y_percussive


# Предварительная обработка аудио с использованием SciPy
def preprocess_audio(file_path, min_length=2048, min_kernel_size=32, default_win_length=512, silence_threshold=1e-5):
    logging.info(f"Начало обработки файла: {file_path}")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Шаг 1. Загрузка аудиофайла с помощью SciPy
    y, sr = load_audio(file_path)
    # Шаг 2. Проверка минимальной длины
    if len(y) < min_length:
        raise ValueError(f"Аудио слишком короткое для обработки: {len(y)} samples (минимум: {min_length} samples)")

    # Шаг 3. Нормализация громкости
    y_normalized = normalize_audio(y)

    # Шаг 4. Удаление низкочастотных помех с помощью низкочастотного фильтра
    y_filtered = lowpass_filter(y_normalized, sr)

    # Проверка амплитуды сигнала
    if np.max(np.abs(y_filtered)) < silence_threshold:
        raise ValueError(f"Сигнал слишком тихий после обработки: max амплитуда = {np.max(np.abs(y_filtered))}")

    return y_filtered, sr


# Пример использования
if __name__ == "__main__":
    input_file = "../downloads/saved_raya_reference_pydub.wav"  # Замените на ваш путь к аудиофайлу
    preprocess_audio(input_file)
