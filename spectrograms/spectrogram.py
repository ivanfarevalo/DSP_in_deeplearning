import numpy as np
import matplotlib.pyplot as plt
import windows as windows
import sys
from scipy.io import wavfile
from scipy import signal
from matplotlib.colors import LogNorm
import sounddevice as sd
from scipy.io.wavfile import write

def record_void(fs, seconds, output_file='output.wav', num_channels=1):
    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=num_channels)
    sd.wait()  # Wait until recording is finished
    write(output_file, fs, myrecording)  # Save as WAV file

class Spectogram():
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs

    def generate_spectogram(self, window_type, win_size, overlap):

        win = windows.Window(win_size)
        window = np.zeros(win_size)
        if window_type == "hamming":
            window = win.hamming_window()
        elif window_type == "rectangular":
            window = win.rectangular_window()
        elif window_type == "triangular":
            window = win.triangular_window()
        elif window_type == "sine":
            window = win.sine_window()
        elif window_type == "hann":
            window = win.hann_window()
        elif window_type == "blackman":
            window = win.blackman_window()

        num_segments = int(1 + (len(self.signal)-win_size)/(win_size*(1-overlap)))
        spectogram = np.zeros((win_size//2 +1, num_segments))
        for i in range(num_segments):
            start_idx = int(win_size*(1-overlap)*i)
            end_idx = int(win_size*(1-overlap)*i + win_size)
            spectogram[:,i] = abs(np.fft.fft(self.signal[start_idx:end_idx]*window, win_size)[:win_size//2 +1])
            # Don't need to convert to dB since LogNorm is being passed to pcolormesh
            # spectogram[:,i] = 10 * np.log10(abs(np.fft.fft(self.signal[start_idx:end_idx]*window, win_size)[:win_size//2 +1])**2 / (self.fs * win_size))

        t = np.linspace(0, len(self.signal) / fs, spectogram.shape[1])
        f = np.linspace(0, int(fs / 2)/1000, spectogram.shape[0])

        return t, f, spectogram

    def plot(self, t, f, spectogram, fig=plt):
        print('inplot')
        fig.pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        fig.ylabel('Frequency [Hz]')
        fig.xlabel('Time [sec]')
        # plt.show()


if __name__ == '__main__':

    print(f"len(sys.argv) args: sys.argv")

    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        print(f"Spectrogram Analysis for file {input_file}")
        fs, data = wavfile.read(input_file)


    if len(sys.argv) > 2:
        fs = int(sys.argv[1])
        audio_length =int(sys.argv[2])
        output_filename = './output.wav'
        if len(sys.argv) == 4:
            output_filename = sys.argv[3]
        elif len(sys.argv) > 4:
            print('Takes at most 3 flags: fs, time to record, and optional: output file name. Default filename: output.wav')
            sys.exit(1)

        print(f"Spectrogram Analysis for file recorded audio")
        record_void(fs, audio_length)
        fs, data = wavfile.read(output_filename)


    #FOR DEBUGGING
    '''
    fs, data = wavfile.read('./speech_data/P501_C_english_m2_IRS_08k.wav')
    fs, data = wavfile.read('./speech_data/Female_1_16k.wav')
    data = data[:len(data)//3] # Focusing on first third of signal
    fs = 16000 # Sample rate
    audio_length = 6  # Duration of recording in seconds
    record_void(fs, audio_length)
    fs, data = wavfile.read('./output.wav')
    '''

    windows_list = ['rectangular', 'hann', 'hamming', 'blackman']
    win_size = [64, 256, 1024]
    p_overlap = [0, 0.25, 0.5]

    spec = Spectogram(data, fs)

    # Experiment 1: compare all windows with 50%  overlap for narrow and wide band
    fig, axs = plt.subplots(2,2)
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, window in enumerate(windows_list):
        t, f, spectogram = spec.generate_spectogram(window, win_size[0], p_overlap[2])
        axs[indices[i]].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[indices[i]].set_ylabel('Frequency [kHz]')
        axs[indices[i]].set_xlabel('Time [sec]')
        axs[indices[i]].set_title(f"64pt-{window} with 50% overlap")
    plt.suptitle("Wide band spectogram")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)
    for i, window in enumerate(windows_list):
        t, f, spectogram = spec.generate_spectogram(window, win_size[2], p_overlap[2])
        axs[indices[i]].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[indices[i]].set_ylabel('Frequency [kHz]')
        axs[indices[i]].set_xlabel('Time [sec]')
        axs[indices[i]].set_title(f"1024pt-{window} with 50% overlap")
    plt.suptitle("Narrow band spectogram")
    plt.tight_layout()
    plt.show()

    # Experiment 2: compare different window sizes for the hamming window at 0% and 50% overlap.
    fig, axs = plt.subplots(3,1)
    indices = [(0, 0), (1,0), (2,0)]
    for i in range(len(win_size)):
        t, f, spectogram = spec.generate_spectogram(windows_list[2], win_size[i], p_overlap[0])
        axs[i].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[i].set_ylabel('Frequency [kHz]')
        axs[i].set_xlabel('Time [sec]')
        axs[i].set_title(f"{win_size[i]}pt-{windows_list[2]} with 0% overlap")
    plt.suptitle("Window Length Comparison")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3, 1)
    indices = [(0, 0), (1, 0), (2, 0)]
    for i in range(len(win_size)):
        t, f, spectogram = spec.generate_spectogram(windows_list[2], win_size[i], p_overlap[2])
        axs[i].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[i].set_ylabel('Frequency [kHz]')
        axs[i].set_xlabel('Time [sec]')
        axs[i].set_title(f"{win_size[i]}pt-{windows_list[2]} with 50% overlap")
    plt.suptitle("Window Length Comparison")
    plt.tight_layout()
    plt.show()

    # Experiment 3: compare different overlaps for the hamming window
    fig, axs = plt.subplots(3, 1)
    indices = [(0, 0), (1, 0), (2, 0)]
    for i in range(len(win_size)):
        t, f, spectogram = spec.generate_spectogram(windows_list[2], win_size[0], p_overlap[i])
        axs[i].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[i].set_ylabel('Frequency [kHz]')
        axs[i].set_xlabel('Time [sec]')
        axs[i].set_title(f"{win_size[0]}pt-{windows_list[2]} with {p_overlap[i]*100}% overlap")
    plt.suptitle("% Overlap Comparison")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3, 1)
    indices = [(0, 0), (1, 0), (2, 0)]
    for i in range(len(win_size)):
        t, f, spectogram = spec.generate_spectogram(windows_list[2], win_size[2], p_overlap[i])
        axs[i].pcolormesh(t, f, spectogram, shading='gouraud', norm=LogNorm())
        axs[i].set_ylabel('Frequency [kHz]')
        axs[i].set_xlabel('Time [sec]')
        axs[i].set_title(f"{win_size[2]}pt-{windows_list[2]} with {p_overlap[i]*100}% overlap")
    plt.suptitle("% Overlap Comparison")
    plt.tight_layout()
    plt.show()

    '''
    Alternative implementation
    
    spec = Spectogram(data, fs)
    t, f, spectogram = spec.generate_spectogram('hamming', win_size, p_overlap)
    spec.plot(t, f, spectogram)

    f, t, Sxx = signal.spectrogram(data, fs, window='hamming', nperseg=win_size, noverlap=int(win_size * p_overlap))
    fig = plt.figure()
    plt.pcolormesh(t, f, Sxx, shading='gouraud', norm=LogNorm())
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    fig2 = plt.figure()
    plt.specgram(data, Fs=fs, window=np.hamming(win_size), NFFT=win_size, noverlap=int(win_size * p_overlap))
    plt.show()
    '''