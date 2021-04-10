# Signal Processing Front Ends in Deep Learning

Building on the success of deep learning for image recognition, the deep learning
approach for one dimensional signals such as speech, audio and biological (EEG and ECG) signals is to
first convert them to two dimensional signals using frequency domain transformations and then apply 2D
image recognition techniques. “End-to-end” deep learning purports to work directly on raw data and avoid
domain specific analyses and feature selection. However, front end signal processing such as the frequency
domain transformations for end-to-end deep learning cannot be avoided and many decisions are made for
common applications, including the particular frequency domain representation, bandwidth, sampling rate,
window type, window length, and window overlap, that significantly impact the results. The aim is to surveys and analyzes these decisions and presents techniques to make informed
choices.

### Experiments:

  1. **spectrograms/windows.py:** analyze different window functions and their spectral characteristics. \
[Link to Report](https://github.com/ivanfarevalo/DSP_in_deeplearning/blob/master/spectrograms/Windowing_report.pdf)

  2. **spectrograms/spectrogram.py:** experiment how different choices of window function, window length, and window overlap affect spectrogram quality and characteristics of audio signals. \
[Link to Report](https://github.com/ivanfarevalo/DSP_in_deeplearning/blob/master/spectrograms/Spectrogram_report.pdf)




# Information Theoretic Techniques in Agent Learning

Using information theoretic tools, we analyze how a learning agent, upon
taking some observations of the environment, develops an understanding of the structure of the
environment, formulates models of this structure, and studies any remaining apparent randomness or
unpredictability. We see how unseen or unmodeled structure can be interpreted as unpredictability and
even randomness. The primary information theoretic tools employed are entropy, entropy rate, relative
entropy, and mutual information. These studies are important since the “black box” nature of deep learning
prevents critical, needed insights into what is being analyzed and what is identified might be incorrect.
