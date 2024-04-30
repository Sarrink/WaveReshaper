import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal

fname_im = "cow.png"
fname_audio = "mooing-cow-122255.wav"

# Calculate desired envelope from image
im = np.array(Image.open(fname_im))
bw = (255 - im).sum(axis=2)
bw = bw > 5

# Remove white background
top = []
btm = []
left = bw.shape[1]
right = 0

for i in range(bw.shape[1]):
    col = bw[:, i]
    nonzero = np.nonzero(col)
    if nonzero[0].size > 0:
        first = np.min(nonzero)
        last = np.max(nonzero)
        col[first:last] = True
        btm.append(last)
        top.append(first)
        left = min(left, i)
        right = i
    else:
        top.append(np.NaN)
        btm.append(np.NaN)

bw = bw[:, left:right]
top = np.array(top[left:right])
btm = np.array(btm[left:right])

top_min = np.min(top)
btm_max = np.max(btm)
mid = (top_min + btm_max) / 2

# Calculate envelope
im_env = np.empty((top.size, 3))

im_env[:, 0] = (mid - top) / (mid - top_min)
im_env[:, 2] = (btm - mid) / (mid - btm_max)
im_env[:, 1] = (im_env[:, 0] + im_env[:, 2]) / 2

im_env_diff = im_env[:, 0] - im_env[:, 1]

plt.imshow(bw)
plt.show()

plt.plot(im_env)
plt.show()

# Load sound file
sr, audio = wavfile.read(fname_audio)

# Convert to mono
audio = audio.sum(axis=1) / audio.shape[1]

# Trim audio
sound = np.nonzero(np.abs(audio) > 100)
sound_first = np.min(sound)
sound_last = np.max(sound)

audio = audio[sound_first:sound_last] / np.max(np.abs(audio))

# Find audio envelope
# Use peaks of peaks as overall shape
w = 25

peaks, _ = signal.find_peaks(audio, width=w)
peaks_peaks, _ = signal.find_peaks(audio[peaks])

min_peaks, _ = signal.find_peaks(-audio, width=w)
min_peaks_peaks, _ = signal.find_peaks(-audio[min_peaks])

# Interpolate peaks of peaks to audio length
x = np.arange(0, audio.size)
audio_env = np.zeros((audio.size, 3))
audio_env[:, 0] = np.interp(x, peaks[peaks_peaks], audio[peaks[peaks_peaks]])
audio_env[:, 2] = np.interp(
    x, min_peaks[min_peaks_peaks], audio[min_peaks[min_peaks_peaks]]
)

plt.plot(audio)
plt.plot(audio_env)
plt.show()

# Resample desired envelope to audio length
im_resample = np.empty((audio.size, 3))

for i in range(3):
    im_resample[:, i] = signal.resample(im_env[:, i], audio.size)

im_diff = signal.resample(im_env_diff, audio.size)

# Change audio amplitude to desired amplitude
new_audio = np.empty(audio.size)

for i in range(audio.size):
    sample = audio[i]
    if sample > 0:
        new_audio[i] = im_resample[i, 1] + (
            sample / np.abs(audio_env[i, 0]) * np.abs(im_diff[i])
        )
        new_audio[i] = min(new_audio[i], im_resample[i, 0])
    else:
        new_audio[i] = im_resample[i, 1] + (
            sample / np.abs(audio_env[i, 2]) * np.abs(im_diff[i])
        )
        new_audio[i] = max(new_audio[i], im_resample[i, 2])

wavfile.write(fname_im + ".wav", sr, new_audio.astype(np.float32))

plt.plot(new_audio)
plt.show()
