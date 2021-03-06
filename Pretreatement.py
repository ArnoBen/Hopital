import os
os.chdir(r'C:\Users\Arno\OneDrive\Documents\Stage Hopital\Code\Python')

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ImportExcel as iex
from scipy import signal as sg
import mne
import MNEfunctions as mnef
from scipy.signal import welch
from scipy.fftpack import fft

patient_number = 205
sample_rate = 250
epoch_duration = 2 #in seconds
NFFT = sample_rate * epoch_duration
filepath = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\eeg.txt"
sig = pd.read_csv(filepath,sep=';',decimal='.', nrows=iex.getMaxRowLimit(patient_number))
fields = sig.columns.values.tolist()

#Filter definition
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(1/fn, 30/fn),'bandpass')
#%% 
#Times extracted from excel
LOC = iex.getPropofolTime(patient_number)
ROC = iex.getWakeUpTime(patient_number)

eeg = np.array(np.ones([31, sig.shape[0]]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]])
sig = None
#Divide eeg into epochs
#Tout d'abord, il faut supprimer quelques points pour rendre le nombre total de points divisible par 500 (sampling freq*2sec)
pts_per_epoch = sample_rate*epoch_duration
reste = eeg.shape[1]%(250*epoch_duration)
eeg = eeg[:, reste:]
#Ensuite on peut réarranger le tableau en epochs de 2 secondes
#n epochs, 31 channels, 2 sec/epochs --> (n, 31, 500)
eeg_epochs = np.reshape(eeg, (31, np.int(eeg.shape[1]/(pts_per_epoch)),  pts_per_epoch)) 
eeg_epochs = np.moveaxis(eeg_epochs, 1, 0)
eeg = None
#Create events : Awake (1), Asleep (2)
event_id = dict(awake_preAG=1, asleep=2, awake_postAG=3)
events = []

true_events = True
# ACTUAL EVENTS
if true_events:
    for i in range(eeg_epochs.shape[0]):
        if i*pts_per_epoch < LOC:
            events.append([i,0,1])
        elif i*pts_per_epoch > ROC:
            events.append([i,0,3])
        else:
            events.append([i,0,2])
#%%
ls_channel = ['FP1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2']#, 'Fz']
#ls_channel = ['E1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 
#              'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
bad_channels = [ls_channel[ch] for ch in iex.getBadChannels(patient_number)]
info = mne.create_info(ch_names=ls_channel, sfreq=sample_rate, ch_types='eeg')
info['bads'] = bad_channels
mtg = mne.channels.read_montage("standard_1020",ch_names=ls_channel, unit='cm')
epochs = mne.EpochsArray(eeg_epochs, info, events, event_id= event_id, tmin=0, reject = dict(eeg=0.3))
epochs.set_montage(mtg, set_dig=False)
#epochs.plot(scalings=dict(eeg=1.5e-1), n_epochs=5, n_channels=31)

LOC, ROC = mnef.getLOCROC(epochs, 'time',epoch_duration*sample_rate)
#%%
#Let's create three separate objects : awake before AG, asleep, awake after AG
#epochs_awake1 = epochs.copy(); epochs_awake1.crop(tmin = 0, tmax = LOC)
#epochs_asleep = epochs.copy(); epochs_asleep.crop(tmin = LOC, tmax = ROC)
#epochs_awake2 = epochs.copy(); epochs_awake2.crop(tmin = ROC)
#%%
#raw = mne.io.RawArray(eeg, info)
#raw.set_montage(mtg, set_dig=False)
#times = raw.times
#epochs = mne.Epochs(raw)#, events, event_id)#, tmin=0, tmax=epoch_duration)
##raw.plot(scalings=dict(eeg=2e-1), n_channels=10)
##raw.plot_projs_topomap(times[250*60*20 : 250*60*30])
#%% ICA
method = 'fastica'

# Choose other parameters
n_components = 20 # if float, select n_components by explained variance of PCA
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 1

ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
#print(ica)
#reject = dict(eeg=5e-2)
ica.fit(epochs)
ica.plot_components(inst=epochs)
#ica.plot_sources(inst=epochs)
#%% Apply correction
#We have to visually inspect the eog component
eog_component = iex.getBadICAs(patient_number)
ica.exclude = eog_component
epochs_corrected = epochs.copy()
ica.apply(epochs_corrected)
epochs_corrected_backup = epochs_corrected.copy()
#epochs_corrected.plot(scalings=dict(eeg=1.5e-1), n_epochs=5, n_channels = 31)
#%% Removing epochs with points > 3*std

#We need to know at which epochs the patient loses consciousness and wakes up:
[LOC_epoch, ROC_epoch] = mnef.getLOCROC(epochs_corrected)
std_channels = [] # (std_values, 3) ,
for i in range(31): 
    std_channels.append([  np.std(epochs_corrected.get_data()[:LOC_epoch,i,:]),          # std pre LOC
                           np.std(epochs_corrected.get_data()[LOC_epoch:ROC_epoch,i,:]), # std asleep
                           np.std(epochs_corrected.get_data()[ROC_epoch:,i,:])           # std post ROC
                        ])                     
#%%
epoch_count = 0
bad_epochs_count = 0
state = 0
bad_epochs = []
bad_epoch_fft = []
for epoch in epochs_corrected:
    if LOC_epoch - 50 < epoch_count < LOC_epoch + 50 : epoch_count += 1 ; continue;
    if ROC_epoch - 50 < epoch_count < ROC_epoch + 50 : epoch_count += 1 ; continue;
    if epoch_count < LOC_epoch : state = 0
    elif epoch_count > LOC_epoch and epoch_count < ROC_epoch : state = 1
    elif epoch_count > ROC_epoch : state = 2
    for i in range(31):
        #If the channel is marked as bad, we ignore it:
        is_bad_ch = False
        for ch in epochs_corrected.info['bads'] :
            if ls_channel[i] == ch : is_bad_ch = True
        if is_bad_ch : continue;
        #Comparison to standard deviation :
        epoch_amplitude = np.max(epoch[i,:]) - np.min(epoch[i,:])
        if epoch_amplitude > 10*std_channels[i][state] or epoch_amplitude > 8 * np.std(epoch[i,:]):
            bad_epochs_count += 1
            bad_epochs.append(epoch_count)
            break
        #Detection of abnormal FFT:
        epoch_fft = np.abs(np.fft.fft(epoch[i,:250]))
        epoch_fft = epoch_fft[:np.int(len(epoch_fft)/2)]
        if np.max(epoch_fft[25:]) > 1.5 or np.max(epoch_fft[:5]) > 6 or np.max(epoch_fft) > 4:
            bad_epochs_count += 1
            bad_epochs.append(epoch_count)
            bad_epoch_fft.append([epoch_count,i])
            break
#       Detection of abnormal psd:
#        f, epoch_psd = welch(epoch[i,:], nfft=epoch_duration*sample_rate)
#        std_epoch_psd = np.std(epoch_psd)
#        if np.max(epoch_psd) > 
    epoch_count += 1
    #if count == 10:break;
print(bad_epochs_count)       
#epochs_corrected.drop(bad_epochs)
#epochs_corrected.plot(scalings=dict(eeg=1.5e-1), n_epochs=5, n_channels = 31)
#epochs_corrected.save(r'C:\Users\Arno\Documents\Patients\Epochs\patient' + str(patient_number) + '-epo-30Hz.fif')
#%%
faulty_epoch = epochs_corrected.get_data()[404,9,:]
plt.figure()

plt.subplot(3,1,1)
plt.plot(faulty_epoch)

faulty_epoch_fft = fft(faulty_epoch)
plt.subplot(3,1,2)
plt.plot(np.abs(faulty_epoch_fft[:30]))

f, faulty_epoch_psd = Pxx = welch(faulty_epoch, nfft=epoch_duration * sample_rate)
plt.subplot(3,1,3)
plt.plot(faulty_epoch_psd[:30])