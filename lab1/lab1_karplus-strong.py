import numpy as np
import scipy.io.wavfile as wave


def GuitarString(frequency, duration=1., sample_rate=44100, toType=False):
    # Сигнал за одну секунду колеблется sample_rate/length раз.
    # Тогда length = sample_rate/frequency.
    noise = np.random.uniform(-1, 1, int(sample_rate / frequency))  # Создаем шум

    samples = np.zeros(int(sample_rate * duration))
    for i in range(len(noise)):
        samples[i] = noise[i]
    for i in range(len(noise), len(samples)):
        # В начале i меньше длины шума, поэтому мы берем значения из шума.
        # Но потом, когда i больше длины шума, мы уже берем посчитанные нами новые значения.
        samples[i] = (samples[i - len(noise)] + samples[i - len(noise) - 1]) / 2

    if toType:
        samples = samples / np.max(np.abs(samples))  # Нормируем от -1 до 1
        return np.int16(samples * 32767)  # Переводим в тип данных int16
    else:
        return samples


while True:
    frequency = float(input("Enter frequency -> "))
    if frequency < 10 : break
    sound = GuitarString(frequency, duration=2, toType=True)
    wave.write(f"SoundGuitarString{frequency}.wav", 44100, sound)
