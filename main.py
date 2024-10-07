import logging
import os

from feature_matching import extract_voice_features, match_segments
from preprocessing import load_audio, extract_mfcc
from utils import get_all_audio_files, timecode_format

# Пути
BASE_DIR = os.path.dirname(__file__)
REFERENCE_VOICE_PATH = os.path.join(BASE_DIR, "downloads", "raya_reference.wav")
AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "E:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт")
OUTPUT_FILE = os.path.join(BASE_DIR, "matching_segments.txt")

# Логирование
logging.basicConfig(filename="search_log.log", level=logging.INFO)


def process_reference(reference_voice_path):
    """Обрабатывает эталонное аудио и возвращает его голосовые признаки."""
    audio, sr = load_audio(reference_voice_path)
    mfcc = extract_mfcc(audio, sr)
    reference_features = extract_voice_features(mfcc)
    return reference_features


def process_audio_files(audio_folder_path, reference_features, output_file):
    """Обрабатывает все аудиофайлы в папке и сохраняет совпадения."""
    audio_files = get_all_audio_files(audio_folder_path)
    with open(output_file, "w") as out_file:
        for audio_file in audio_files:
            logging.info(f"Processing file: {audio_file}")
            matches = match_segments(reference_features, audio_file)
            for start, end, similarity in matches:
                line = f"File: {audio_file}, Start: {timecode_format(start)}, End: {timecode_format(end)}, Similarity: {similarity:.2f}\n"
                out_file.write(line)
                logging.info(line)
            logging.info(f"Completed processing: {audio_file}")


if __name__ == "__main__":
    # Обработка эталонного голоса
    reference_features = process_reference(REFERENCE_VOICE_PATH)

    # Обработка всех аудиофайлов и сохранение совпадений
    process_audio_files(AUDIO_FOLDER_PATH, reference_features, OUTPUT_FILE)
    print(f"Все совпадения сохранены в {OUTPUT_FILE}")
