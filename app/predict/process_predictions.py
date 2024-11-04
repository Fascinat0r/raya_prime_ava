import matplotlib.pyplot as plt
import pandas as pd

from config import Config  # Импорт конфигурационного файла
from preprocessing.find_silence import detect_silence_intervals


def process_predictions(file_path, config):
    try:

        # Загружаем предсказания
        data = pd.read_csv(file_path)

        # Проверка наличия нужного столбца
        if 'prediction' not in data.columns or 'start_time' not in data.columns:
            raise KeyError("Файл должен содержать столбцы 'prediction' и 'start_time'.")

        # Применяем скользящее среднее к предсказаниям
        window_size = config.MOVING_AVERAGE_WINDOW
        data['smoothed_prediction'] = data['prediction'].rolling(window=window_size, min_periods=1).mean()

        # Применяем пороговое значение вероятности
        data['is_voice'] = data['smoothed_prediction'] >= config.THRESHOLD

        # Инициализация переменных для хранения объединенных интервалов
        intervals = []
        current_start = None

        for i, row in data.iterrows():
            if row['is_voice']:
                if current_start is None:
                    # Начало нового интервала
                    current_start = row['start_time']
            else:
                if current_start is not None:
                    # Завершаем текущий интервал
                    current_end = data.iloc[i - 1]['start_time']
                    intervals.append((current_start, current_end))
                    current_start = None

        # Если есть незавершенный интервал, добавляем его
        if current_start is not None:
            intervals.append((current_start, data.iloc[-1]['start_time']))

        # Фильтруем короткие интервалы
        intervals = [(start, end) for start, end in intervals if end - start >= config.MIN_DURATION]

        # Объединяем близко расположенные интервалы
        merged_intervals = []
        for start, end in intervals:
            if merged_intervals and start - merged_intervals[-1][1] <= config.MERGE_GAP:
                # Объединяем с предыдущим интервалом
                merged_intervals[-1] = (merged_intervals[-1][0], end)
            else:
                # Добавляем новый интервал
                merged_intervals.append((start, end))

        return data, intervals, merged_intervals

    except KeyError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None, None, None


def plot_predictions(data, intervals, merged_intervals, adjusted_intervals, silence_intervals):
    # Создаем график
    plt.figure(figsize=(15, 8))

    # Оригинальные предсказания
    plt.plot(data['start_time'], data['prediction'], label="Original Predictions", color="blue", alpha=0.6)

    # Сглаженные предсказания
    plt.plot(data['start_time'], data['smoothed_prediction'], label="Smoothed Predictions", color="orange", alpha=0.8)

    # Отмечаем начальные и конечные интервалы детекции голоса
    for start, end in intervals:
        plt.axvspan(start, end, color="green", alpha=0.3, label="Voice Intervals (Raw)")

    # Объединенные интервалы
    for start, end in merged_intervals:
        plt.axvspan(start, end, color="red", alpha=0.5, label="Voice Intervals (Merged)")

    # Интервалы тишины
    for start, end in silence_intervals:
        plt.axvspan(start, end, color="gray", alpha=0.5, label="Silence Intervals", zorder=0)

    # Скорректированные интервалы
    for start, end in adjusted_intervals:
        plt.axvspan(start, end, color="purple", alpha=0.5, label="Voice Intervals (Adjusted)")

    plt.xlabel("Time (s)")
    plt.ylabel("Prediction")
    plt.title("Voice Activity Detection with Smoothing and Interval Merging")

    # Уникальная легенда
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.grid(True)
    plt.show()


def adjust_intervals_to_silence(voice_intervals, silence_intervals, left_tolerance=0.5, right_tolerance=0.2):
    """
    Корректирует интервалы голосовых данных на основе ближайших интервалов тишины.

    :param voice_intervals: Список кортежей (start_time, end_time) с предсказанными интервалами голоса.
    :param silence_intervals: DataFrame с интервалами тишины (колонки 'start_time', 'end_time').
    :param left_tolerance: Максимальное расстояние до ближайшего интервала тишины слева для корректировки начала интервала (в секундах).
    :param right_tolerance: Максимальное расстояние до ближайшего интервала тишины справа для корректировки конца интервала (в секундах).
    :return: Скорректированный список интервалов
    """
    adjusted_intervals = []
    silence_starts = silence_intervals['start_time'].values
    silence_ends = silence_intervals['end_time'].values

    for start, end in voice_intervals:
        # Найти ближайший интервал тишины к началу голосового интервала
        nearest_start_silence = silence_ends[(silence_ends <= start) & (start - silence_ends <= left_tolerance)]

        # Найти ближайший интервал тишины к концу голосового интервала
        nearest_end_silence = silence_starts[(silence_starts >= end) & (silence_starts - end <= right_tolerance)]

        # Корректировка начала интервала
        adjusted_start = nearest_start_silence[-1] if nearest_start_silence.size > 0 else start

        # Корректировка конца интервала
        adjusted_end = nearest_end_silence[0] if nearest_end_silence.size > 0 else end

        # Добавляем скорректированный интервал
        adjusted_intervals.append((adjusted_start, adjusted_end))

    return adjusted_intervals


# Пример использования

if __name__ == "__main__":
    config = Config()
    file_path = "predictions.csv"
    data, intervals, merged_intervals = process_predictions(file_path, config)
    if data is None:
        print("Ошибка при обработке предсказаний.")

    silence_intervals = detect_silence_intervals("output_segment.wav")

    adjusted_intervals = adjust_intervals_to_silence(merged_intervals, silence_intervals)

    # DataFrame silence_intervals (N,2) to dict ((start_time,  end_time))
    silence_intervals = [tuple(x) for x in silence_intervals.to_numpy()]

    plot_predictions(data, intervals, merged_intervals, adjusted_intervals, silence_intervals)

    # Сохраняем обработанные интервалы в файл CSV
    with open("processed_intervals.csv", "w") as f:
        f.write("start_time,end_time\n")
        for start, end in adjusted_intervals:
            f.write(f"{start},{end}\n")
