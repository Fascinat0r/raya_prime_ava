import os
import subprocess


# Функция для извлечения аудиодорожки из видеофайлов
def extract_audio_from_mp4(folder_path):
    # Проход по всем файлам в указанной папке
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_file = os.path.join(folder_path, filename)
            # Генерация имени для аудиофайла (с тем же названием, но расширением .wav)
            audio_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.wav")
            # Проверка, существует ли уже аудиофайл
            if os.path.exists(audio_file):
                print(f"Файл {audio_file} уже существует. Пропуск.")
                continue
            # Вызов ffmpeg для извлечения аудиодорожки
            subprocess.run([
                "ffmpeg", "-i", video_file,  # Входной файл
                "-vn",  # Отключение видеопотока
                "-acodec", "pcm_s16le",  # Сохранение в исходном качестве (16-бит PCM)
                "-ar", "44100",  # Частота дискретизации (44.1 кГц)
                "-y",  # Перезапись выходного файла без подтверждения
                audio_file  # Имя выходного файла
            ], check=True)


# Пример использования
video_folder = "E:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт"  # Путь к папке с видеофайлами
extract_audio_from_mp4(video_folder)
