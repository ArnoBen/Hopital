import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from mne import EvokedArray, EpochsArray, create_info
import mne

from sklearn.decomposition import FastICA, PCA

plt.ion()

#fname = "eeg.txt"
#sig = pd.read_csv("eeg.txt", sep=';', decimal=".")

eeg = np.array(sig.ix[:, 4:])
sig = None

# centering of the eeg signal and filtering
# filter of 0.6*Nyquist_freq
b, a = signal.butter(5, (0.032, 0.052), 'bandpass')  # get only alpha band
b, a = signal.butter(5, (2. / 250, 30 / 250),
                     'bandpass')  # get only alpha band
for c in range(32):
    sig = np.array(eeg[:, c])
    # sig = signal.filtfilt(b, a, sig)
    # sig -= pd.rolling_mean(sig, 500)
    # sig[:2000] = 0
    # sig *= abs(sig) < 0.25
    sig[1400000:1600000] = 0
    eeg[:, c] = sig

# plt.plot(np.arange(len(eeg)) * 1./500, eeg[:, 0])
# plt.show()

# Montage of easycap
ls_channel = ['FP1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'FP2']

ls_channel = ['FP1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'Fz']
mtg = mne.channels.read_montage(
    "standard_1005", ch_names=ls_channel, unit='cm')

# Creation of an evoke object
info = create_info(ls_channel, 500, montage=mtg, ch_types="eeg")

# # Compute ICA
# ica = FastICA(n_components=32)
# S_ = ica.fit_transform(eeg[1200000:2000000])  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix
# # We can `prove` that the ICA model applies by reverting the unmixing.
# assert np.allclose(eeg[1200000:2000000], np.dot(S_, A_.T) + ica.mean_)

S_ = eeg[3300 * 500:3500 * 500]  # [1270000:1310000]
evoked = EvokedArray(S_.T, info)
evoked.info['bads'] = ["FC1", "CP5", "CP1"]  # drop of bad channels
evoked.set_eeg_reference(["Fz"])  # set the ref
# evoked.drop_channels(["F8", "FC2", "FC6"])  # drop of bad channels

# evoked.drop_channels(['FP1', 'F7', 'F3', 'Fz', 'F8', 'FT9', 'FC1', 'FC2', 'FC6', 'FT10', 'TP10', 'P7', 'O1', 'Oz'])  # drop of a bad channel
# evoked.set_eeg_reference(["FC5"])  # set the ref

times = np.arange(0, 300, 0.01)
# If times is set to None only 10 regularly spaced topographies will be shown
# plot magnetometer data as topomaps
for i in range(0, 5):
    evoked.plot_topomap(times[i * 10:i * 10 + 10], ch_type="eeg",
                        proj=True, show_names=True, vmin=-10000, vmax=10000)

# ts_args = dict(gfp=True)
# evoked.plot_joint(times=times, ts_args=ts_args)

# Correlation entre les capteurs
OMEGA = []
freqbins = np.linspace(0, 500, len(S_[:, 0]))
for i in range(31):
    if not i == 5:
        spectrum = np.fft.fft(S_[:, i])
        magnitude = np.abs(spectrum)
        # plt.plot(freqbins, magnitude)
        phase = np.unwrap(np.angle(spectrum))
        OMEGA.append(phase)
        # plt.plot(freqbins, phase, label=i)

plt.figure()
plt.imshow(np.corrcoef(np.array(OMEGA)))
plt.colorbar()

# Meme chose pour les spectrogrammes
S_ = eeg
spectro = []
for i in range(32):
    A = plt.specgram(S_[:, i], NFFT=1024, Fs=500)[0]
    spectro.append(np.sum(A, axis=0))
spectro = np.array(spectro)
spectro = (spectro.T / np.std(spectro, axis=1)).T
spectro[-1] = 0

evoked = EvokedArray(spectro, info)
evoked.info['bads'] = ["FC1", "CP5", "CP1"]  # drop of bad channels
evoked.set_eeg_reference(["Fz"])  # set the ref

times = evoked.times
for i in range(200, 205):
    evoked.plot_topomap(times[i * 10:i * 10 + 10],
                        ch_type="eeg", show_names=True)
evoked.plot()
