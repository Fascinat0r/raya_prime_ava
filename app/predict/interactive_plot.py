# app/predict/interactive_plot.py
# Description: Визуализация результатов предсказаний с выделением обработанных интервалов используя Plotly.

import pandas as pd
import plotly.graph_objects as go


def visualize_predictions(predictions_csv, intervals_csv, output_html="prediction_visualization.html"):
    """
    Визуализирует результаты предсказаний с выделением обработанных интервалов.

    :param predictions_csv: Путь к CSV файлу с предсказаниями (prediction, start_time).
    :param intervals_csv: Путь к CSV файлу с обработанными интервалами (start_time, end_time).
    :param output_html: Путь для сохранения HTML файла с интерактивной визуализацией.
    """
    # Загружаем данные
    predictions_df = pd.read_csv(predictions_csv)
    intervals_df = pd.read_csv(intervals_csv)

    # Создаем интерактивный график
    fig = go.Figure()

    # Добавляем линию для предсказаний
    fig.add_trace(go.Scatter(
        x=predictions_df["start_time"],
        y=predictions_df["prediction"],
        mode="lines+markers",
        name="Prediction",
        line=dict(color="blue"),
        marker=dict(size=5)
    ))

    # Добавляем прямоугольники для каждого обработанного интервала
    for _, row in intervals_df.iterrows():
        fig.add_shape(
            type="rect",
            x0=row["start_time"],
            x1=row["end_time"],
            y0=predictions_df["prediction"].min(),
            y1=predictions_df["prediction"].max(),
            fillcolor="lightgreen",
            opacity=0.3,
            line_width=0,
            layer="below"
        )

    # Настраиваем дизайн графика
    fig.update_layout(
        title="Interactive Prediction Visualization with Processed Intervals",
        xaxis_title="Time (seconds)",
        yaxis_title="Prediction Value",
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True), title="Time (seconds)"),
        yaxis=dict(title="Prediction"),
        hovermode="x unified"
    )

    # Сохраняем график как HTML
    fig.write_html(output_html)
    print(f"Интерактивный график сохранен в {output_html}")

    # Для отображения графика в браузере при запуске
    fig.show()


# Пример использования
if __name__ == "__main__":
    predictions_csv = "predictions.csv"  # Путь к вашему CSV файлу с предсказаниями
    intervals_csv = "processed_intervals.csv"  # Путь к вашему CSV файлу с интервалами
    visualize_predictions(predictions_csv, intervals_csv)
