import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from app.train.cross_entropy.cross_entropy_model import MelSpecCrossEntropyNet
from app.train.pt_utils import restore_model
from config import Config
from logger import get_logger
from preprocessing.full_processing_pipeline import process_audio_to_spectrograms

logger = get_logger("predict")


def predict_audio_segments(model, device, mel_spectrograms):
    """
    Применяет модель для предсказания по каждому сегменту мел-спектрограммы.
    :param model: Модель для предсказания.
    :param device: Устройство (CPU или GPU).
    :param mel_spectrograms: Список мел-спектрограмм для предсказания.
    :return: Список вероятностей для каждого сегмента.
    """
    model.eval()
    predictions = []
    start_times = []

    for start_time, mel_spectrogram in mel_spectrograms:
        mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)  # Добавляем размерность batch
        with torch.no_grad():
            output = model(mel_spectrogram)
            probabilities = F.softmax(output, dim=1)  # Применяем softmax к логитам
            class_1_prob = probabilities[0][1].item()  # Вероятность для класса "1"
            predictions.append(class_1_prob)
            start_times.append(start_time)

    return predictions, start_times


def plot_predictions(predictions):
    """
    Отображает график предсказаний по сегментам аудио и обработанных интервалов.
    :param predictions: Массив предсказаний для каждого сегмента.
    """
    plt.figure(figsize=(18, 6))

    # График исходных предсказаний
    plt.plot(predictions, label="Исходные предсказания", color="blue", alpha=0.6)

    plt.xlabel("Временная метка сегмента (секунды)")
    plt.ylabel("Класс предсказания")
    plt.title("Исходные предсказания")
    plt.grid(True)
    plt.savefig("predictions_with_intervals.png")
    plt.show()


def main(audio_file, model_path, config):
    """
    Основная функция для обработки аудио файла и применения модели для предсказания.
    :param audio_file: Путь к аудио файлу.
    :param model_path: Путь к сохраненной модели.
    :return: Обработанные интервалы.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info(f"Используемое устройство: {device}")

    model = MelSpecCrossEntropyNet(reduction='mean').to(device)
    model = restore_model(model, model_path, device)

    logger.info(f"Обработка аудио файла: {audio_file}")
    mel_spectrograms = process_audio_to_spectrograms(audio_file, config, overlap=0.9)

    logger.info("Применение модели к сегментам...")
    predictions, start_times = predict_audio_segments(model, device, mel_spectrograms)

    logger.info(f"Предсказания для аудио файла {audio_file}: {predictions}")

    # Выводим график предсказаний и обработанных интервалов
    plot_predictions(predictions)

    return predictions, start_times


if __name__ == "__main__":
    # Пример использования
    audio_file = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #14 ХАКЕР ГРАБИТЕЛЬ • Майнкрафт.wav"
    model_path = "../train/weights/"  # Путь к папке с моделью
    config = Config()
    predictions, start_times = main(audio_file, model_path, config)
    segment_time = config.HOP_LENGTH / 44100

    # Сохраняем предсказания в файл csv
    with open("predictions.csv", "w") as f:
        f.write("prediction,start_time,end_time\n")
        for predictions, start_times in zip(predictions, start_times):
            f.write(f"{predictions},{start_times},{start_times + segment_time}\n")
