import numpy as np
import matplotlib.pyplot as plt


class Window():
    def __init__(self, N):
        self.N = N
        self.window = np.arange(self.N)

    def rectangular_window(self):
        return np.ones(self.N)

    def triangular_window(self):
        window = ( (self.N - 1)/2 - np.abs(self.window - (self.N -1)/2) ) * 2/(self.N -1)
        return window

    def sine_window(self):
        window = np.sin(np.pi * self.window / (self.N -1))
        return window

    def hann_window(self):
        window = 1/2 * (1 - np.cos(2*np.pi*self.window / (self.N -1) ) )
        return window

    def hamming_window(self):
        window = 0.54 - 0.46 * np.cos(2*np.pi*self.window / (self.N -1) )
        return window

    def blackman_window(self):
        window = 0.42 - 0.5 * np.cos(2*np.pi*self.window / (self.N -1) ) + 0.08*np.cos(4*np.pi*self.window / (self.N -1) )
        return window


if __name__ == '__main__':

    # Generate windows
    N_odd = 31 # Triangular window_even
    N_even = 32 # All other windows
    window_even = Window(N_even)
    window_odd = Window(N_odd)
    rectangular_window = window_even.rectangular_window()
    hann_window = window_even.hann_window()
    sine_window = window_even.sine_window()
    triangular_window = window_odd.triangular_window()
    hamming_window = window_odd.hamming_window()
    blackman_window = window_odd.blackman_window()

    # Plotting windows on same graph.
    fig = plt.figure()
    plt.plot(rectangular_window, label='Rectangular Window')
    plt.plot(hann_window, '--', label='Hann Window')
    plt.plot(sine_window, '*-', label='Sine Window')
    plt.plot(triangular_window, '.-', label='Triangular Window')
    plt.legend()
    plt.title("Windows"), plt.xlabel('Time')

    fig2 = plt.figure()
    plt.plot(hamming_window, '--', label='Hamming Window')
    plt.plot(blackman_window, label='Blackman Window')
    plt.legend()
    plt.title("Windows"), plt.xlabel('Time')
    plt.show()


    # Plot normalized magnitude spectrum for each window
    num_fft_pts = 1024
    rectangular_window_fft = np.fft.fftshift(np.fft.fft(rectangular_window, num_fft_pts))
    hann_window_fft = np.fft.fftshift(np.fft.fft(hann_window, num_fft_pts))
    sine_window_fft = np.fft.fftshift(np.fft.fft(sine_window, num_fft_pts))
    triangular_window_fft = np.fft.fftshift(np.fft.fft(triangular_window, num_fft_pts))
    hamming_window_fft = np.fft.fftshift(np.fft.fft(hamming_window, num_fft_pts))
    blackman_window_fft = np.fft.fftshift(np.fft.fft(blackman_window, num_fft_pts))
    norm_freq = np.linspace(-0.5, 0.5, num_fft_pts)

    fig3, axs = plt.subplots(2,2)
    axs[0, 0].plot(norm_freq, abs(rectangular_window_fft)), axs[0, 0].set_title('Rectangular Window')
    axs[0, 1].plot(norm_freq, abs(hann_window_fft)), axs[0, 1].set_title('Hann Window')
    axs[1, 0].plot(norm_freq, abs(sine_window_fft)), axs[1, 0].set_title('Sine Window')
    axs[1, 1].plot(norm_freq, abs(triangular_window_fft)), axs[1, 1].set_title('Triangular Window')
    plt.suptitle("Normalized magnitude spectra")
    fig3.tight_layout()
    plt.show()

    fig4, axs = plt.subplots(1, 2)
    axs[0].plot(norm_freq, abs(hamming_window_fft)), axs[0].set_title('Hamming Window')
    axs[1].plot(norm_freq, abs(blackman_window_fft)), axs[1].set_title('Blackman Window')
    plt.suptitle("Normalized magnitude spectra")
    fig4.tight_layout()
    plt.show()

    # Plot normalized dB spectrum of rectangular and triangular windows
    fig5 = plt.figure()
    plt.plot(norm_freq, 20*np.log10( abs(rectangular_window_fft)/max( abs(rectangular_window_fft) ) ), label='Rectangular Window')
    plt.plot(norm_freq, 20*np.log10( abs(triangular_window_fft)/max( abs(triangular_window_fft) ) ), '--', label='Triangular Window')
    plt.legend()
    plt.title("Normalized dB spectra"), plt.xlabel('Normalized Frequency')
    plt.xlim(-0.2, 0.2), plt.ylim(-80, 5)
    plt.show()

    # Plot normalized dB spectrum of rectangular and triangular windows
    fig6 = plt.figure()
    plt.plot(norm_freq, 20 * np.log10(abs(sine_window_fft) / max(abs(sine_window_fft))), label='Sine Window')
    plt.plot(norm_freq, 20 * np.log10(abs(hann_window_fft) / max(abs(hann_window_fft))), '--', label='Hann Window')
    plt.legend()
    plt.title("Normalized dB spectra"), plt.xlabel('Normalized Frequency')
    plt.xlim(-0.2, 0.2), plt.ylim(-80, 5)
    plt.show()

    # Plot normalized dB spectrum of rectangular and triangular windows
    fig7 = plt.figure()
    plt.plot(norm_freq, 20 * np.log10(abs(hamming_window_fft) / max(abs(hamming_window_fft))), label='Hamming Window')
    plt.plot(norm_freq, 20 * np.log10(abs(blackman_window_fft) / max(abs(blackman_window_fft))), '--', label='Blackman Window')
    plt.legend()
    plt.title("Normalized dB spectra"), plt.xlabel('Normalized Frequency')
    plt.xlim(-0.2, 0.2), plt.ylim(-80, 5)
    plt.show()





