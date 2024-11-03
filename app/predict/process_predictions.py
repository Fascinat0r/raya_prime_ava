import pandas as pd
import numpy as np
from scipy.signal import medfilt


def process_predictions(file_path, threshold=0.5, min_duration=0.5, merge_gap=0.1):
    try:
        # Загружаем предсказания
        data = pd.read_csv(file_path)

        # Проверка наличия нужного столбца
        if 'prediction' not in data.columns:
            raise KeyError("В файле отсутствует столбец 'prediction'. Проверьте заголовки файла.")

        # Применяем медианное сглаживание
        data['smoothed_prediction'] = medfilt(data['prediction'], kernel_size=3)

        # Применяем пороговое значение вероятности
        data['is_voice'] = data['smoothed_prediction'] >= threshold

        # Инициализация переменных для хранения объединенных интервалов
        intervals = []
        current_start = None
        current_end = None

        for _, row in data.iterrows():
            if row['is_voice']:
                if current_start is None:
                    # Начало нового интервала
                    current_start = row['start_time']
                current_end = row['end_time']
            else:
                if current_start is not None:
                    # Завершаем текущий интервал и сохраняем его
                    intervals.append((current_start, current_end))
                    current_start = None
                    current_end = None

        # Если есть незавершенный интервал, добавляем его
        if current_start is not None:
            intervals.append((current_start, current_end))

        # Фильтруем короткие интервалы
        intervals = [(start, end) for start, end in intervals if end - start >= min_duration]

        # Объединяем близко расположенные интервалы
        merged_intervals = []
        for start, end in intervals:
            if merged_intervals and start - merged_intervals[-1][1] <= merge_gap:
                # Объединяем с предыдущим интервалом
                merged_intervals[-1] = (merged_intervals[-1][0], end)
            else:
                # Добавляем новый интервал
                merged_intervals.append((start, end))

        return merged_intervals

    except KeyError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Пример использования
processed_intervals = process_predictions("predictions.csv")
print("Обработанные интервалы:", processed_intervals)

# Сохраняем обработанные интервалы в файл csv
with open("processed_intervals.csv", "w") as f:
    f.write("start_time,end_time\n")
    for start, end in processed_intervals:
        f.write(f"{start},{end}\n")
