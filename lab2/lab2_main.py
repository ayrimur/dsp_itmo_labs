import librosa
import numpy as np
import scipy.signal, scipy.fft
import soundfile as sf
import matplotlib.pyplot as plt

'''
Реализовать вычисление дискретного преобразования Фурье для типовых после-
довательностей (единичный импульс, единичный скачок, синусоидальное колебание).
'''
"""

def fourier_transform(x):
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.matmul(x, M)


imp = np.zeros(128)
imp[0] = 1

my_imp_fourier = fourier_transform(imp)
plt.plot(np.abs(my_imp_fourier))
plt.title("Мой фурье, единичный импульс")
plt.show()

imp_fourier = scipy.fft.fft(imp)
plt.plot(np.abs(imp_fourier))
plt.title("Scipy фурье, единичный импульс")
plt.show()


jump = np.zeros(128)
jump[63:] = 1

my_jump_fourier = fourier_transform(jump)
plt.plot(np.abs(my_jump_fourier))
plt.title("Мой фурье, скачок")
plt.show()

jump_fourier = scipy.fft.fft(jump)
plt.title("Scipy фурье, скачок")
plt.plot(np.abs(jump_fourier))
plt.show()


sin = np.sin(np.arange(128))
my_sin_fourier = fourier_transform(sin)
plt.title("Мой фурье, синус")
plt.plot(np.abs(my_sin_fourier))
plt.show()

sin_fourier = scipy.fft.fft(sin)
plt.title("Scipy фурье, синус")
plt.plot(np.abs(sin_fourier))
plt.show()

"""
'''
С использованием дискретного преобразования Фурье проанализировать спектральный 
состав сигнала паровозного гудка. Построить амплитудный спектр сигнала. Определить 
на каких частотах расположены три основные гармоники сигнала. 
'''

whistle, sr = librosa.load('train_whistle.wav', 44100)
N = whistle.shape[0]
whistle_fourier = np.fft.fft(whistle)[:N//2]  # np.fft.fftshift()

x_sine_wave_spectrum = np.linspace(0, sr / 2, len(whistle_fourier))

whistle_A = np.abs(whistle_fourier)


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(whistle_A)
axs[0].set_xlabel('Время, сек')
axs[0].set_ylabel('Амплитуда гудка')


axs[1].plot(x_sine_wave_spectrum, np.arctan2(np.imag(whistle_fourier), np.real(whistle_fourier)))
axs[1].set_xlabel('Частота, Гц')
axs[1].set_ylabel('Фаза гудка')
plt.subplots_adjust(hspace=0.5)
plt.show()


f = np.linspace(0, len(whistle) - 1, len(whistle))
f *= sr / len(whistle)

A_maxs = np.sort(-whistle_A[:whistle_A.size // 2])[:15]
f_max_idxs = np.argsort(-whistle_A[:whistle_A.size // 2])[:15]
print("Сигнал гудка: ", f[f_max_idxs])

'''
С использованием дискретного преобразования Фурье / оконного преобразования Фурье 
построить амплитудный спектр / спектрограмму сигнала первого спутника (04.10.1957, СССР). 
'''
signal, sr = librosa.load('sputnik_1.wav')
N = signal.shape[0]

fourier_image = scipy.fft.fft(signal, workers=8)

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axs[0].plot(np.arange(N) / sr, signal)
axs[0].set_xlabel('Время, сек')
axs[0].set_ylabel('Амплитуда спутника')

axs[1].plot(np.angle(fourier_image)[:N // 2], linewidth=0.5)
axs[1].set_xlabel('Частота, Гц')
axs[1].set_ylabel('Фаза спутника')

f, t, S = scipy.signal.spectrogram(signal, sr, window='hamming', return_onesided=True, mode='magnitude')

axs[2].pcolormesh(t, f, S, shading='gouraud')
axs[2].set_xlabel('Время, сек')
axs[2].set_ylabel('Спектрогамма')
plt.subplots_adjust(hspace=0.5)
plt.show()


"""
'''
Проанализировать с использованием оконного преобразования Фурье двухтональный 
многочастотный сигнал (Dual-Tone Multi-Frequency, DTMF). Определить «номер телефона» 
(порядок набора цифр) закодированный в нём. 
'''
dtmf, sr = librosa.load('dtmf.wav', 44100)
f, t, S = scipy.signal.spectrogram(dtmf, sr, window='hamming', return_onesided=True, mode='magnitude')
plt.pcolormesh(t, f, S, shading='gouraud')
plt.ylabel('Частота, Гц')
plt.xlabel('Время, сек')
plt.title("Спектрограмма DTMF")
plt.show()


one_digit_len = sr * 3
first_digit_A = np.abs(scipy.fft.fft(dtmf[0:one_digit_len]))
second_digit_A = np.abs(scipy.fft.fft(dtmf[one_digit_len:one_digit_len * 2]))
third_digit_A = np.abs(scipy.fft.fft(dtmf[one_digit_len * 2:one_digit_len * 3]))

f = np.linspace(0, one_digit_len - 1, one_digit_len)

A_maxs = np.sort(-first_digit_A[:first_digit_A.size // 2])[:2]
f_max_idxs = np.argsort(-first_digit_A[:first_digit_A.size // 2])[:2]
print("первая цифра:", f[f_max_idxs] * sr / len(first_digit_A))

f = np.linspace(0, one_digit_len - 1, one_digit_len)

A_maxs = np.sort(-second_digit_A[:second_digit_A.size // 2])[:2]
f_max_idxs = np.argsort(-second_digit_A[:second_digit_A.size // 2])[:2]
print("вторая цифра:", f[f_max_idxs] * sr / len(second_digit_A))

f = np.linspace(0, one_digit_len - 1, one_digit_len)

A_maxs = np.sort(-third_digit_A[:third_digit_A.size // 2])[:2]
f_max_idxs = np.argsort(-third_digit_A[:third_digit_A.size // 2])[:2]
print("третья цифра:", f[f_max_idxs] * sr / len(third_digit_A))


'''
Требуется ответить на вопрос: «Какой спектр сигнала (амплитудный или фазовый) 
более информативен?» Для ответа на вопрос требуется вычислить амплитудный и фазовый 
спектры сигнала, а затем выполнить реконструкцию сигнала, используя обратное 
преобразование Фурье, двумя способами: только по амплитудному спектру, обнулив 
фазовый; только по фазовому спектру, положив амплитудный равным единице. Прослушать 
полученные сигнала и сделать соответствующие выводы. 
'''
my_voice, sr = librosa.load('voice.wav', 44100)
my_voice_fourier = scipy.fft.fft(my_voice)
sf.write("voice_from_amplitude.wav", np.abs(scipy.fft.ifft(np.abs(my_voice_fourier))), sr)
my_voice_fourier.real = 1
sf.write("voice_from_phase.wav", np.abs(scipy.fft.ifft(np.arctan2(my_voice_fourier.imag, my_voice_fourier.real))), sr)

"""
'''
Для некоторого речевого сигнала реализовать алгоритм построения мел-частотных 
кепстральных коэффициентов (Mel-Frequency Cepstral Coefficients, MFCCs).
'''

from skimage.util import view_as_windows
from functools import partial

def _compute_filterbank(n_mfcc, win_len, sample_rate):

    # Declare convertation functions
    freq_to_mel = lambda x: 2595.0 * np.log10(1.0 + x / 700.0)
    mel_to_freq = lambda x: 700 * (10 ** (x / 2595.0) - 1.0)

    # Compute filterbank markup
    mel_min = freq_to_mel(0)
    mel_max = freq_to_mel(sample_rate)
    mels = np.linspace(mel_min, mel_max, n_mfcc)
    freqs = mel_to_freq(mels)
    filter_points = np.floor(freqs * (win_len // 2 + 1) / sample_rate).astype(np.int)

    # Compute filters
    filters = np.zeros([filter_points.shape[0], int(win_len / 2 + 1)])
    for i in range(1, filter_points.shape[0] - 1):
        filters[i, filter_points[i - 1]: filter_points[i]] = np.linspace(0, 1, filter_points[i] - filter_points[i - 1])
        filters[i, filter_points[i]: filter_points[i + 1]] = np.linspace(1, 0, filter_points[i + 1] - filter_points[i])

    filters = filters[:, :-1]
    return filters


def mfcc(n_mfcc, signal, sample_rate, win_len, win_step, window_function, use_dct=True):
    # Split into frames and apply window function
    frames = view_as_windows(signal, window_shape=(win_len,), step=win_step)
    w_funcs = {'hanning': np.hanning, 'hamming': np.hamming, 'bartlett': np.bartlett,
               'kaiser': partial(np.kaiser, beta=3), 'blackman': np.blackman}
    frames = frames * w_funcs[window_function](win_len + 1)[:-1]

    # Compute power spectrum (periodogram)
    frames = frames.T
    spectrum = scipy.fft.fft(frames, axis=0, workers=8)[:int(win_len / 2)]
    spectrum = np.flip(spectrum, axis=0)
    power_spectrum = np.abs(spectrum) ** 2

    # Compute mel-filterbank
    filterbank = _compute_filterbank(n_mfcc, win_len, sample_rate)
    print(filterbank.shape)

    # Apply filterbank
    filtered_spectrum = np.dot(filterbank, power_spectrum).T
    log_spectrum = 10.0 * np.log10(filtered_spectrum)
    log_spectrum = log_spectrum[:, 1:-1]

    # Extract mfcc using dct-II
    mfcc = scipy.fft.dct(log_spectrum, type=2, n=n_mfcc, workers=-1)

    return mfcc

signal, sr = librosa.load('human_speech.wav', 44100)

coefs = mfcc(
    n_mfcc=23,
    signal=signal,
    sample_rate=sr,
    win_len=2048,
    win_step=512,
    window_function='hamming',
    use_dct=False
).T
'''plt.figure(figsize=(10, 5))
plt.title("23 MFCC (My)")
plt.imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
plt.show()

# Extract mfcc using librosa api
coefs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=23)
plt.figure(figsize=(10, 5))
plt.title("23 MFCC (librosa)")
plt.imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
plt.show()'''

print(coefs)

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
#axs[0].figure(figsize=(10, 5))
axs[0].imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
axs[0].set_ylabel('My MFCC')

coefs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=23)
#axs[1].figure(figsize=(10, 5))
axs[1].imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
axs[1].set_ylabel('23 MFCC (librosa)')
plt.subplots_adjust(hspace=0.5)
plt.show()
