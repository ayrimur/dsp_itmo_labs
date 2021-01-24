from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift


"""
Любыми известными способами осуществить построение импульсной и переходной 
характеристик, амплитудно-частотной и фазочастотной характеристик КИХ- и 
БИХ-фильтров первого порядка.

Осуществить обработку типовых последовательностей (единичный импульс, 
единичный скачок, синусоидальное колебание) анализируемыми фильтрами.
Сделать соответствующие выводы по полученным результатам.
"""

# sin wave with noise
f = 10
fs = 200
t = np.arange(0, 20, 1 / fs)
sig = np.sin(2 * np.pi * f * t)
noise = np.random.normal(1, 0.5, len(t))
sig = sig + noise
sig_spectrum = abs(fftshift(fft(sig)))
x_sig_spectrum = np.linspace(-fs / 2, fs / 2, len(sig_spectrum))

# finite impulse response filter
nyq_rate = fs / 2.0
cutoff_freq = 11
beta = 5
N = 35
taps = signal.firwin(N, cutoff_freq / nyq_rate, window=('kaiser', beta))

plt.suptitle('КИХ фильтр с окном Казера-Бесселя', fontsize=14)
plt.subplot(3, 1, 1)
plt.plot(taps)
plt.title('Импульсная х-ка')
plt.grid(True)

w, h = signal.freqz(taps)
plt.subplot(3, 1, 2)
plt.title('Амплитудная х-ка')
plt.plot((w / np.pi) * nyq_rate, abs(h))
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title('Фазовая х-ка')
plt.plot((w / np.pi) * nyq_rate, np.arctan2(np.imag(h), np.real(h)))
plt.grid(True)
plt.tight_layout()
plt.show()

# signal filtered with fir-filter
filtered_sig = signal.lfilter(taps, 1, sig)
filtered_sig_spectrum = abs(fftshift(fft(filtered_sig)))

plt.suptitle('Сигнал с КИХ фильтром', fontsize=14)
plt.subplot(2, 2, 1)
plt.title('Исходный сигнал')
plt.plot(t, sig)
plt.xlim(10, 11)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.title('Спектр исходного сигнала')
plt.plot(x_sig_spectrum, sig_spectrum)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.title('Фильтрированный сигнал')
plt.plot(t, filtered_sig)
plt.xlim(10, 11)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.title('Спектр фильтрированного сигнала')
plt.plot(x_sig_spectrum, filtered_sig_spectrum)
plt.grid(True)
plt.tight_layout()
plt.show()


# infinite impulse response filter
cutoff_freq = 11
b, a = signal.butter(3, cutoff_freq / nyq_rate)

plt.suptitle('Фильтр Баттерворта', fontsize=14)
plt.subplot(3, 1, 1)
plt.plot(b)
plt.title('Импульсная х-ка')
plt.grid(True)

w, h = signal.freqz(b, a)
plt.subplot(3, 1, 2)
plt.title('Амплитудная х-ка')
plt.plot((w / np.pi) * nyq_rate, abs(h))
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title('Фазовая х-ка')
plt.plot((w / np.pi) * nyq_rate, np.arctan2(np.imag(h), np.real(h)))
plt.grid(True)
plt.tight_layout()
plt.show()

# signal filtered with iir-filter
filtered_sig = signal.lfilter(b, a, sig)
filtered_sig_spectrum = abs(fftshift(fft(filtered_sig)))

plt.suptitle('Сигнал с БИХ фильтром', fontsize=14)
plt.subplot(2, 2, 1)
plt.title('Исходный сигнал')
plt.plot(t, sig)
plt.xlim(10, 11)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.title('Спектр исходного сигнала')
plt.plot(x_sig_spectrum, sig_spectrum)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.title('Фильтрированный сигнал')
plt.plot(t, filtered_sig)
plt.xlim(10, 11)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.title('Спектр фильтрированного сигнала')
plt.plot(x_sig_spectrum, filtered_sig_spectrum)
plt.grid(True)
plt.tight_layout()
plt.show()
