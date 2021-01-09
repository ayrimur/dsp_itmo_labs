import librosa
import numpy as np
import scipy.signal, scipy.fft
import soundfile as sf
import matplotlib.pyplot as plt

'''
Реализовать вычисление дискретного преобразования Фурье для типовых после-
довательностей (единичный импульс, единичный скачок, синусоидальное колебание).
'''


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


'''
С использованием дискретного преобразования Фурье проанализировать спектральный 
состав сигнала паровозного гудка. Построить амплитудный спектр сигнала. Определить 
на каких частотах расположены три основные гармоники сигнала. 
'''

whistle, sr = librosa.load('train_whistle.wav', 44100)
whistle_fourier = scipy.fft.fft(whistle)

whistle_A = np.abs(whistle_fourier)
whistle_Ph = np.arctan2(whistle_fourier.imag, whistle_fourier.real)
plt.plot(whistle_A)
plt.title("Амплитудный спектр гудка")
plt.show()
plt.plot(whistle_Ph)
plt.title("Фазовый спектр гудка")
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
sputnik_1, sr = librosa.load('sputnik_1.wav', 44100)
sputnik_1_fourier = scipy.fft.fft(sputnik_1)

sputnik_1_A = np.abs(sputnik_1_fourier)
plt.plot(sputnik_1_A)
plt.title("Амплитудный спектр спутника")
plt.show()

sputnik_1_Ph = np.arctan2(sputnik_1_fourier.imag, sputnik_1_fourier.real)
plt.plot(sputnik_1_Ph)
plt.title("Фазовый спектр спутника")
plt.show()

f, t, S = scipy.signal.spectrogram(sputnik_1, sr, window='hamming', return_onesided=True, mode='magnitude')
plt.pcolormesh(t, f, S, shading='gouraud')
plt.ylabel('Частота, Гц')
plt.xlabel('Время, сек')
plt.title("Спектрограмма спутника")
plt.show()

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


'''
Для некоторого речевого сигнала реализовать алгоритм построения мел-частотных 
кепстральных коэффициентов (Mel-Frequency Cepstral Coefficients, MFCCs).
'''


def frame_audio(audio, FFT_size=2048, hop_size=10, sr=44100):
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sr * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))
    for i in range(frame_num):
        frames[i] = audio[i * frame_len:i * frame_len + FFT_size]
    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = mel_to_freq(mels)
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points) - 2, FFT_size // 2 + 1))

    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])
    return filters


human_speech, sr = librosa.load('human_speech.wav', 44100)

human_speech_framed = frame_audio(human_speech, FFT_size=2048, hop_size=10, sr=sr)
window = scipy.signal.get_window("hamming", 2048, fftbins=True)
human_speech_framed *= window
human_speech_framed_ = np.transpose(human_speech_framed)

audio_fft = np.empty((1 + 2048 // 2, human_speech_framed_.shape[1]), dtype=np.complex64, order='F')

for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = scipy.fft.fft(human_speech_framed_[:, n], axis=0)[:audio_fft.shape[0]]

audio_fft = np.transpose(audio_fft)
audio_fft_power = np.square(np.abs(audio_fft))

filter_points, mel_freqs = get_filter_points(0, sr / 2, 10, 2048, 44100)
filters = get_filters(filter_points, 2048)

audio_filtered = np.dot(filters, np.transpose(audio_fft_power))
audio_log = np.log10(audio_filtered)


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
    return basis


plt.plot(np.dot(dct(10, 10), audio_log)[0])  # cepstral coefficients
plt.title("MFCC")
plt.show()
