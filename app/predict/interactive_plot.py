import pandas as pd
import plotly.graph_objects as go


def visualize_predictions(predictions_csv, output_html="prediction_visualization.html"):
    """
    Визуализирует результаты предсказаний с возможностью интерактивного приближения.

    :param predictions_csv: Путь к CSV файлу с предсказаниями (prediction, start_time).
    :param output_html: Путь для сохранения HTML файла с интерактивной визуализацией.
    """
    # Загружаем данные
    predictions_df = pd.read_csv(predictions_csv)

    # Сортируем по времени
    # predictions_df = predictions_df.sort_values("start_time")

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

    # Настраиваем дизайн графика
    fig.update_layout(
        title="Interactive Prediction Visualization",
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
    predictions_csv = "predictions.csv"  # Путь к вашему CSV файлу
    visualize_predictions(predictions_csv)
