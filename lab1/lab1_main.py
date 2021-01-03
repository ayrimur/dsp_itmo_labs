import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft
from scipy.signal import butter, lfilter

if __name__ == '__main__':

    data, sr = librosa.load('poem.wav', sr=44100)
    plt.plot(data)
    plt.title("Исходный сигнал")
    plt.show()

    #  sf.write('poem_22050.wav', librosa.resample(data, sr, 22050), sr)
    #  sf.write('poem_11025.wav', librosa.resample(data, sr, 11025), sr)

    freq = 3*100 + 500
    lowPassFilter = (np.arange(512) * sr / 1024 < freq).astype(float)
    lowPassFilter = np.concatenate((lowPassFilter, lowPassFilter[::-1]))

    plt.plot(lowPassFilter)
    plt.title("Фильтр")
    plt.show()

    filterImpulseResponse = ifft(lowPassFilter)

    plt.plot(filterImpulseResponse.real)
    plt.title("Импульсная х-ка фильтра")
    plt.show()


    def window_stack(a, stepsize=1, width=3):
        n = a.shape[0]
        return np.vstack(a[i:i + width] for i in range(0, n - width + 1, stepsize))

    def convolve(s, f):
        return np.fromiter(
            (s[max(0, i - f.size):i] @ f[::-1][-i:] for i in range(1, s.size + 1)),
            np.float
        )

    filtered = convolve(data, filterImpulseResponse.real[:512])

    plt.plot(filtered)
    plt.title("Filtered")
    plt.show()

    sf.write('poem_filtered.wav', filtered, sr)


  import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft
from scipy.signal import butter, lfilter

if __name__ == '__main__':
    
    data, sr = librosa.load('poem.wav', sr=44100)
    plt.plot(data)
    plt.title("Исходный сигнал")
    plt.show()

    #  sf.write('poem_22050.wav', librosa.resample(data, sr, 22050), sr)
    #  sf.write('poem_11025.wav', librosa.resample(data, sr, 11025), sr)

    freq = 3*100 + 500
    lowPassFilter = (np.arange(512) * sr / 1024 < freq).astype(float)
    lowPassFilter = np.concatenate((lowPassFilter, lowPassFilter[::-1]))

    plt.plot(lowPassFilter)
    plt.title("Фильтр")
    plt.show()

    filterImpulseResponse = ifft(lowPassFilter)

    plt.plot(filterImpulseResponse.real)
    plt.title("Импульсная х-ка фильтра")
    plt.show()

    
    def window_stack(a, stepsize=1, width=3):
        n = a.shape[0]
        return np.vstack(a[i:i + width] for i in range(0, n - width + 1, stepsize))

    def convolve(s, f):
        return np.fromiter(
            (s[max(0, i - f.size):i] @ f[::-1][-i:] for i in range(1, s.size + 1)),
            np.float
        )

    filtered = convolve(data, filterImpulseResponse.real[:512])

    # Reverberation
    plt.plot(filtered)
    plt.title("Filtered")
    plt.show()

    sf.write('poem_filtered.wav', filtered, sr)

    raw_ir, _ = librosa.load('cath_IR.wav')
    plt.plot(raw_ir)
    plt.title('castle')
    plt.show()

    rev_signal = convolve(data, raw_ir)
    plt.plot(rev_signal)
    plt.title('Reverberation')
    plt.show()

    sf.write("holl_acoustic.wav", rev_signal, sr)

