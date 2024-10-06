import librosa
import noisereduce as nr


# Предварительная обработка аудио
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path)

    # 1. Удаление шума
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # 2. Нормализация громкости
    y_normalized = librosa.util.normalize(y_denoised)

    # 3. Удаление низкочастотных помех
    y_filtered = librosa.effects.preemphasis(y_normalized)

    # 4. Разделение вокала и музыки
    y_harmonic, _ = librosa.decompose.hpss(y_filtered)

    return y_harmonic, sr
