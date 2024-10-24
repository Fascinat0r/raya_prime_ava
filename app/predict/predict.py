import torch
from matplotlib import pyplot as plt

from app.train.cross_entropy.cross_entropy_model import MelSpecCrossEntropyNet
from app.train.pt_utils import restore_model
from logger import get_logger
from preprocessing.full_processing_pipeline import process_audio_to_spectrograms

logger = get_logger("predict")


def predict_audio_segments(model, device, mel_spectrograms):
    """
    Применяет модель для предсказания по каждому сегменту мел-спектрограммы.
    :param model: Модель для предсказания.
    :param device: Устройство (CPU или GPU).
    :param mel_spectrograms: Список мел-спектрограмм для предсказания.
    :return: Список предсказаний для каждого сегмента.
    """
    model.eval()
    predictions = []

    for mel_spectrogram in mel_spectrograms:
        mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)  # Добавляем размерность batch
        with torch.no_grad():
            output = model(mel_spectrogram)
            pred = torch.argmax(output, dim=1)
            predictions.append(pred.item())

    return predictions


def plot_predictions(predictions):
    """
    Отображает график предсказаний по сегментам аудио.
    :param predictions: Массив предсказаний для каждого сегмента.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predictions', marker='o')
    plt.xlabel("Audio Segment Index")
    plt.ylabel("Prediction (Class)")
    plt.title("Predictions by Audio Segments")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(audio_file, model_path):
    """
    Основная функция для обработки аудио файла и применения модели для предсказания.
    :param audio_file: Путь к аудио файлу.
    :param model_path: Путь к сохраненной модели.
    :return: Массив предсказаний.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info(f"Используемое устройство: {device}")

    model = MelSpecCrossEntropyNet(reduction='mean').to(device)
    model = restore_model(model, model_path, device)

    logger.info(f"Обработка аудио файла: {audio_file}")
    mel_spectrograms = process_audio_to_spectrograms(audio_file)

    logger.info("Применение модели к сегментам...")
    predictions = predict_audio_segments(model, device, mel_spectrograms)

    logger.info(f"Предсказания для аудио файла {audio_file}: {predictions}")

    # Выводим график предсказаний
    plot_predictions(predictions)

    return predictions


if __name__ == "__main__":
    # Пример использования
    audio_file = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #10 ЖИВОЙ РОБОТ • Майнкрафт.wav"  # Укажите путь к вашему аудио файлу
    model_path = "../weights/"  # Путь к папке с моделью

    predictions = main(audio_file, model_path)
    print("Предсказания:", predictions)
