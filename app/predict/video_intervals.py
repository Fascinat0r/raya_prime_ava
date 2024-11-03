import pandas as pd
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_scale_image(prediction, width=100, height=500):
    """
    Создает изображение шкалы на основе значения предсказания.
    :param prediction: Значение предсказания в диапазоне [0, 1].
    :param width: Ширина шкалы.
    :param height: Высота шкалы.
    :return: Изображение шкалы как массив numpy.
    """
    # Создаем изображение шкалы
    scale_image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(scale_image)

    # Высота заполненной части шкалы
    filled_height = int(prediction * height)

    # Отрисовка заполненной части
    draw.rectangle([0, height - filled_height, width, height], fill="green")

    # Добавляем текст с текущим значением предсказания
    text = f"{prediction:.2f}"
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)  # Получаем координаты ограничивающего прямоугольника текста
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = max(height - filled_height - text_height - 5, 5)
    draw.text((text_x, text_y), text, fill="white", font=font)

    return np.array(scale_image)


def add_prediction_scale_to_video(video_path, predictions_csv, output_path="output_with_scale.mp4"):
    """
    Добавляет шкалу предсказания к видео и сохраняет результат.
    :param video_path: Путь к исходному видеофайлу.
    :param predictions_csv: Путь к CSV файлу с предсказаниями.
    :param output_path: Путь для сохранения нового видео с наложенной шкалой.
    """
    # Загружаем предсказания
    predictions = pd.read_csv(predictions_csv)

    # Загружаем видео
    video = VideoFileClip(video_path)

    # Функция для создания шкалы на каждый кадр
    def make_frame(get_frame, t):
        # Находим предсказание для текущего времени
        current_prediction = predictions.loc[
            (predictions['start_time'] <= t) &
            (predictions['start_time'] > t - 1 / video.fps), 'prediction'
        ].values

        prediction_value = current_prediction[0] if len(current_prediction) > 0 else 0.0

        # Создаем изображение шкалы и преобразуем его в ImageClip для MoviePy
        scale_img = create_scale_image(prediction_value)
        scale_clip = ImageClip(scale_img).set_duration(video.duration)

        return scale_clip.get_frame(t)

    # Создаем клип для шкалы
    scale_clip = video.fl(make_frame).set_position((0, "center")).set_duration(video.duration)

    # Объединяем исходное видео с наложенной шкалой
    final_video = CompositeVideoClip([video, scale_clip.set_position((0, "center"))])

    # Сохраняем результат
    final_video.write_videofile(output_path, codec="libx264")

# Пример использования
if __name__ == "__main__":
    video_path = "D:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт\\Lp. Идеальный МИР #14 ХАКЕР ГРАБИТЕЛЬ • Майнкрафт.mp4"
    predictions_csv = "predictions.csv"
    add_prediction_scale_to_video(video_path, predictions_csv)
