# app/utils/mel_to_audio.py
# Description: Восстановление аудио из мел-спектрограммы.
import matplotlib.pyplot as plt
import torchaudio
from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim

# Загружаем аудио WAV файл
waveform, sample_rate = torchaudio.load("data/raw/example.wav")

# Преобразование в мел-спектрограмму
mel_spectrogram_transform = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,  # Увеличенное разрешение частоты
    hop_length=256,  # Меньший шаг для более точного восстановления
    n_mels=128  # Увеличено количество мел-фильтров
)

# Применяем преобразование к сигналу
mel_spectrogram = mel_spectrogram_transform(waveform)

# Вывод мел-спектрограммы
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram.log2()[0, :, :].detach().cpu().numpy(), cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram")
plt.show()

# Обратное преобразование мел-спектрограммы в линейную спектрограмму
inverse_mel_scale = InverseMelScale(n_stft=1025, n_mels=128)
spectrogram = inverse_mel_scale(mel_spectrogram)

# Восстановление аудио из спектрограммы с использованием большего числа итераций
griffin_lim = GriffinLim(n_fft=2048, hop_length=256, n_iter=60)  # Увеличено количество итераций
reconstructed_waveform = griffin_lim(spectrogram)

# Сохраняем восстановленный сигнал в файл
torchaudio.save("reconstructed_high_quality.wav", reconstructed_waveform, sample_rate)

# Воспроизведение оригинального и восстановленного сигнала
print("Аудио восстановлено и сохранено как 'reconstructed_high_quality.wav'.")
