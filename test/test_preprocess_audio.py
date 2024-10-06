import soundfile as sf

from config import REFERENCE_VOICE_PATH, PROCESSED_AUDIO_PATH
from preprocessing import preprocess_audio


def test_preprocessing(reference_file, output_file):
    print(f"Применение препроцессинга к файлу: {reference_file}")
    try:
        processed_audio, sr = preprocess_audio(reference_file)
        # Сохранение обработанного файла для визуальной проверки
        sf.write(output_file, processed_audio, sr)
        print(f"Обработанный файл сохранен: {output_file}")
    except RuntimeError as e:
        print(f"Ошибка во время предварительной обработки: {e}")


# Основной тест
if __name__ == "__main__":
    test_preprocessing(REFERENCE_VOICE_PATH, PROCESSED_AUDIO_PATH)
