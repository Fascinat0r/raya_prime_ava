# config.py
import os

# Убедитесь, что все пути являются абсолютными
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_VOICE_PATH = os.path.join(BASE_DIR, "downloads", "saved_raya_reference_pydub.wav")
PROCESSED_AUDIO_PATH = os.path.join(BASE_DIR, "output", "processed_raya_reference.wav")
AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "E:\\4K_Video_Downloader\\Lp  Идеальный Мир · Майнкрафт")
MFCC_SAVE_PATH = os.path.join(BASE_DIR, "cache", "saved_reference_mfcc.pkl")
OUTPUT_FILE = os.path.join(BASE_DIR, "matching_segments.txt")
THRESHOLD = 50  # Пороговое значение для поиска совпадений
