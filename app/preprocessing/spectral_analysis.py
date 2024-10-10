import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def plot_spectral_analysis(file_paths: list, output_image: str = None):
    """
    Выполняет спектральный анализ нескольких аудиофайлов и строит их спектрограммы на одном графике с легендой.

    Аргументы:
    file_paths (list): Список путей к входным аудиофайлам (.wav).
    output_image (str): Путь для сохранения спектрограммы в виде изображения (по умолчанию None).
    """

    plt.figure(figsize=(14, 8))  # Размер итогового графика

    # Обработка каждого файла из списка
    for file_path in file_paths:
        # Загружаем аудиофайл
        audio_data, sample_rate = sf.read(file_path)

        # Преобразование в моно, если многоканальное аудио
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Преобразование сигнала в мел-спектрограмму
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)

        # Логарифмическое масштабирование амплитуды
        S_db = librosa.power_to_db(S, ref=np.max)

        # Вычисление среднего значения спектра по временной оси
        mean_spectrum = np.mean(S_db, axis=1)

        # Построение среднего спектра для каждого файла с легендой
        plt.plot(mean_spectrum, label=f"{file_path}")

    # Оформление графика
    plt.xlabel("Частота (Мел-шкала)")
    plt.ylabel("Амплитуда (дБ)")
    plt.title("Средний спектр нескольких аудиофайлов")
    plt.legend(loc='upper right')  # Добавление легенды в правый верхний угол
    plt.grid(True)

    # Сохранение изображения, если задано
    if output_image:
        plt.savefig(output_image)
    else:
        plt.show()


if __name__ == "__main__":
    input_files = ["data/normalized/example.wav", "data/denoised/example.wav"]
    plot_spectral_analysis(input_files)
