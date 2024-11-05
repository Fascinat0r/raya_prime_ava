import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip


def create_scale_image(prediction, width=100, height=500):
    """
    Создает изображение шкалы на основе значения предсказания.
    :param prediction: Значение предсказания в диапазоне [0, 1].
    :param width: Ширина шкалы.
    :param height: Высота шкалы.
    :return: Изображение шкалы как массив numpy.
    """
    scale_image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(scale_image)

    filled_height = int(prediction * height)
    draw.rectangle([0, height - filled_height, width, height], fill="green")

    text = f"{prediction:.2f}"
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
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
        # Находим ближайшее предсказание по времени
        predictions['time_diff'] = abs(predictions['start_time'] - t)
        nearest_prediction = predictions.loc[predictions['time_diff'].idxmin(), 'prediction']

        # Создаем изображение шкалы и преобразуем его в ImageClip для MoviePy
        scale_img = create_scale_image(nearest_prediction)
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
    video_path = "data/output_segment.mp4"
    predictions_csv = "predictions.csv"
    add_prediction_scale_to_video(video_path, predictions_csv)
