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

patient_number = 225
filepath = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\eeg.txt"
sig = pd.read_csv(filepath,sep=';',decimal='.')

# Si 500Hz, downsample 250Hz
if 0.95 <= sig.iloc[500,0] - sig.iloc[0,0] <= 1.05 : #Si le point 500 est à quasiment 1 seconde
    sig = sig.iloc[::2,:]
fields = sig.columns.values.tolist()    

# Filtering
sample_rate = 250

fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(1/fn, 50/fn),'bandpass')

#%%
plt.close('all')
fig = plt.figure(figsize=(20,10), num='Patient ' + str(patient_number))
ax = fig.subplots(2,2)

SedationStartTime = iex.getPropofolTime(patient_number)
SedationStopTime = iex.getSedationStopTime(patient_number)
WakeUpTime = iex.getWakeUpTime(patient_number)
epoch_size = 2500

#Réveillé avant AG
eeg = np.array(np.ones([31, epoch_size]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]][250*60*4:250*60*4 + epoch_size])
CorrMatrix = np.corrcoef(eeg)
plotting.plot_matrix(CorrMatrix, labels=fields[4:35],  title='Awake pre-GA', axes=ax[0][0], vmin=-1, vmax = 1)

#Endormi
eeg = np.array(np.ones([31, epoch_size]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]][SedationStartTime + 250*60*5 : SedationStartTime + 250*60*5 + epoch_size]) #15min seulmenent pour éviter bistouri elec
CorrMatrix = np.corrcoef(eeg)
plotting.plot_matrix(CorrMatrix, labels=fields[4:35],  title='Asleep', axes=ax[0][1], vmin=-1, vmax = 1)

#Réveillé après AG
eeg = np.array(np.ones([31, epoch_size]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]][WakeUpTime + 250*60*20 : WakeUpTime + 250*60*20 + epoch_size]) #5min - 25min post réveil (bcp artefacts pdt déplacements)
CorrMatrix = np.corrcoef(eeg)
plotting.plot_matrix(CorrMatrix, labels=fields[4:35],  title='Awake post-GA', axes=ax[1][0], vmin=-1, vmax = 1)