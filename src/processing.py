import librosa
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

np.set_printoptions(suppress=True)

y, sample_rate = librosa.load('../heat_waves.mp3', offset=50, duration=3)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

short_term_fourier_transformation = librosa.stft(y)  # STFT of y

D_harmonic, D_percussive = librosa.decompose.hpss(short_term_fourier_transformation, margin=16)


# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(short_term_fourier_transformation))

amplitude = librosa.amplitude_to_db(D_percussive, ref=rp)
hist_min = amplitude.min()
hist_max = amplitude.max()
# for second in amplitude:
#     # print(second.max() if second.max() < -20 else 'low')
#     histogram = np.histogram(second, bins='auto', range=(hist_min, hist_max))[0]
#     print(histogram)


def map_amplitude_to_histogram(values):
    histogram = np.histogram(values, bins=14, range=(hist_min, hist_max))[0]
    mapped_value = map(lambda value: 1 if value else 0, histogram)
    return np.fromiter(mapped_value, int)


result = np.apply_along_axis(map_amplitude_to_histogram, 1, amplitude)
print(result)



# plt.figure(figsize=(12, 8))
#
# plt.subplot(3, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(short_term_fourier_transformation, ref=rp), y_axis='log')
# plt.colorbar()
# plt.title('Full spectrogram')
#
# plt.subplot(3, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
# plt.colorbar()
# plt.title('Harmonic spectrogram')
#
# plt.subplot(3, 1, 3)
# librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
# plt.colorbar()
# plt.title('Percussive spectrogram')
# plt.tight_layout()

# plt.show()
