import pickle

import librosa


# Извлечение MFCC и дополнительных признаков
def extract_mfcc(audio_data):
    try:
        y, sr = audio_data
        # Проверка на слишком короткие сегменты
        if len(y) < 2048:
            raise ValueError("Аудио слишком короткое для извлечения MFCC.")

        # Извлечение MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc
    except Exception as e:
        raise RuntimeError(f"Ошибка при извлечении MFCC: {e}")


# Сохранение MFCC в файл
def save_mfcc(mfcc, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(mfcc, f)


# Загрузка MFCC из файла
def load_mfcc(load_path):
    with open(load_path, 'rb') as f:
        mfcc = pickle.load(f)
    return mfcc
