import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ImportExcel as iex
import math
import os
import pathlib
from matplotlib.pyplot import subplot
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from scipy import signal as sg
from nilearn import plotting
import mne

patient_number = 225
filepath = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\eeg.txt"
sig = pd.read_csv(filepath,sep=';',decimal='.')
sample_rate = 250
fields = sig.columns.values.tolist()
#eeg = sig.iloc[:,4:]
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(1/fn, 48/fn),'bandpass')
#%%

eeg = np.array(np.ones([31, sig.shape[0]]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]])

#Divide eeg into epochs
#Tout d'abord, il faut supprimer quelques points pour rendre le nombre total de points divisible par 500 (sampling freq*2sec)
epoch_duration = 2 #in seconds
reste = eeg.shape[1]%(250*epoch_duration)
eeg = eeg[:, reste:]
#Ensuite on peut réarranger le tableau en epochs de 2 secondes
#31 channels, n epochs, 2 sec/epochs --> (31, n, 500)
eeg_epochs = np.reshape(eeg, (31, np.int(eeg.shape[1]/(sample_rate*epoch_duration)), sample_rate*epoch_duration)) 

#%%
ls_channel = ['FP1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2']#, 'Fz']
info = mne.create_info(ch_names=ls_channel, sfreq=sample_rate, ch_types='eeg')
mtg = mne.channels.read_montage("standard_1020",ch_names=ls_channel, unit='cm')

raw = mne.io.RawArray(eeg, info)
raw.set_montage(mtg, set_dig=False)
times = raw.times
raw.plot_projs_topomap(times[250*60*20 : 250*60*30])
