import logging
import os

from config import REFERENCE_VOICE_PATH, AUDIO_FOLDER_PATH, MFCC_SAVE_PATH, OUTPUT_FILE, THRESHOLD
from feature_extraction import extract_mfcc, load_mfcc, save_mfcc
from matching import find_matching_segments
from preprocessing import preprocess_audio

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # 1. Проверка и подготовка эталонных данных
    if not os.path.exists(MFCC_SAVE_PATH):
        logging.info("Извлечение и сохранение эталонного MFCC...")
        reference_audio = preprocess_audio(REFERENCE_VOICE_PATH)
        reference_mfcc = extract_mfcc(reference_audio)
        save_mfcc(reference_mfcc, MFCC_SAVE_PATH)
    else:
        logging.info("Загрузка ранее сохраненных эталонных данных...")
        reference_mfcc = load_mfcc(MFCC_SAVE_PATH)

    # 2. Поиск совпадений по аудиофайлам в папке
    find_matching_segments(reference_mfcc, AUDIO_FOLDER_PATH, OUTPUT_FILE, threshold=THRESHOLD)


if __name__ == '__main__':
    main()
