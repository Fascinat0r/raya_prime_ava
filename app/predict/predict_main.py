# app/predict/predict_main.py
# Description: Основной скрипт для предсказания интервалов речи в аудиофайле.
from pandas import DataFrame

from config import Config
from predict.prediction import make_prediction
from predict.process_predictions import process_predictions, adjust_intervals_to_silence, plot_predictions
from preprocessing.find_silence import detect_silence_intervals


def predict_main(audio_path, config):
    predictions, start_times = make_prediction(audio_file, config)

    data = DataFrame({'prediction': predictions, 'start_time': start_times})

    data, intervals, merged_intervals = process_predictions(data, config)
    if data is None:
        print("Ошибка при обработке предсказаний.")

    silence_intervals = detect_silence_intervals(audio_path)

    adjusted_intervals = adjust_intervals_to_silence(merged_intervals, silence_intervals)

    # DataFrame silence_intervals (N,2) to dict ((start_time,  end_time))
    silence_intervals = [tuple(x) for x in silence_intervals.to_numpy()]

    plot_predictions(data, intervals, merged_intervals, adjusted_intervals, silence_intervals)

    return predictions, start_times, adjusted_intervals


if __name__ == "__main__":
    audio_file = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #64 ТАЙНЫЙ КУРЬЕР • Майнкрафт.wav"
    model_path = "../train/weights/"  # Путь к папке с моделью
    config = Config()
    predictions, start_times, adjusted_intervals = predict_main(audio_file, config)

    # Сохраняем обработанные интервалы в файл CSV
    with open("data/processed_intervals.csv", "w") as f:
        f.write("start_time,end_time\n")
        for start, end in adjusted_intervals:
            f.write(f"{start},{end}\n")

    # Сохраняем предсказания в файл csv
    with open("data/predictions.csv", "w") as f:
        f.write("prediction,start_time\n")
        for predictions, start_times in zip(predictions, start_times):
            f.write(f"{predictions},{start_times}\n")
