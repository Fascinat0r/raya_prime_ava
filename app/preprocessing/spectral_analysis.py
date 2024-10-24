import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram", xlabel="Time (frames)", ylabel="Mel Bands"):
    """
    Визуализирует мел-спектрограмму.

    Аргументы:
    mel_spectrogram (Tensor или np.array): Массив или тензор мел-спектрограммы размером (1, 64, 64).
    title (str): Заголовок графика.
    xlabel (str): Метка для оси X.
    ylabel (str): Метка для оси Y.
    """
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.cpu().numpy()

    # Выведем предупреждение, если размерность не соответствует ожидаемой
    if mel_spectrogram.shape != (128, 128):
        print(f"Размер мел-спектрограммы не соответствует ожидаемому: {mel_spectrogram.shape}")

    # Если тензор имеет форму (1, 64, 64), убираем ось с единичной размерностью
    if mel_spectrogram.shape[0] == 1:
        mel_spectrogram = mel_spectrogram.squeeze(0)

    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def load_and_visualize_spectrogram(pt_file):
    """
    Загружает мел-спектрограмму из .pt файла и визуализирует её.

    Аргументы:
    pt_file (str): Путь к .pt файлу с сохранённой мел-спектрограммой.
    """
    mel_spectrogram = torch.load(pt_file)  # Загрузка тензора с мел-спектрограммой
    plot_mel_spectrogram(mel_spectrogram, title=f"Mel Spectrogram from {pt_file}")


# Пример использования:
if __name__ == "__main__":
    # Путь к .pt файлу
    pt_file_path = "../data/spectrograms/spectrogram_1000.pt"

    # Визуализация мел-спектрограммы
    for i in range(5):
        path = pt_file_path.replace("1000", str(1000 + i))
        load_and_visualize_spectrogram(path)
