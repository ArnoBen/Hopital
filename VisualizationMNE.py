import os
os.chdir(r'C:\Users\Arno\OneDrive\Documents\Stage Hopital\Code\Python')

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ImportExcel as iex
from scipy import signal as sg
import mne

patient_number = 229
sample_rate = 250
filepath = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\eeg.txt"
sig = pd.read_csv(filepath,sep=';',decimal='.', nrows=iex.getMaxRowLimit(patient_number))
fields = sig.columns.values.tolist()

#Filter definition
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(1/fn, 48/fn),'bandpass')
#%% Times extracted from excel
LOC = iex.getPropofolTime(patient_number)
ROC = iex.getWakeUpTime(patient_number)
#%%
eeg = np.array(np.ones([31, sig.shape[0]]))
for i in range(4,35):
    eeg[i-4] = sg.filtfilt(b,a,sig[fields[i]])

#Divide eeg into epochs
#Tout d'abord, il faut supprimer quelques points pour rendre le nombre total de points divisible par 500 (sampling freq*2sec)
epoch_duration = 3 #in seconds
pts_per_epoch = sample_rate*epoch_duration
reste = eeg.shape[1]%(250*epoch_duration)
eeg = eeg[:, reste:]
#Ensuite on peut réarranger le tableau en epochs de 2 secondes
#n epochs, 31 channels, 2 sec/epochs --> (n, 31, 500)
eeg_epochs = np.reshape(eeg, (31, np.int(eeg.shape[1]/(pts_per_epoch)),  pts_per_epoch)) 
eeg_epochs = np.moveaxis(eeg_epochs, 1, 0)
#Create events : Awake (1), Asleep (2)
event_id = dict(awake=1, asleep=2)
events = []

true_events = True
# ACTUAL EVENTS
if true_events:
    for i in range(eeg_epochs.shape[0]):
        if i*pts_per_epoch < LOC or i*pts_per_epoch > ROC:
            events.append([i,0,1])
        else:
            events.append([i,0,2])
# BULLSHIT EVENTS TO MAKE MNE COMPLY
else:
    state = True
    for i in range(eeg_epochs.shape[1]):
        events.append([i,0,int(state)])
        state = not state
#%%
ls_channel = ['FP1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2']#, 'Fz']
#ls_channel = ['E1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 
#              'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
bad_channels = [ls_channel[ch] for ch in iex.getBadChannels(patient_number)]
info = mne.create_info(ch_names=ls_channel, sfreq=sample_rate, ch_types='eeg')
info['bads'] = bad_channels
mtg = mne.channels.read_montage("standard_1020",ch_names=ls_channel, unit='cm')
#epochs = mne.EpochsArray(eeg_epochs[int(ROC/pts_per_epoch): int(ROC/pts_per_epoch) + 160,:,:], info)#, events, tmin=0, event_id)
epochs = mne.EpochsArray(eeg_epochs, info, events, event_id= event_id, tmin=0, reject = dict(eeg=0.3))
epochs.set_montage(mtg, set_dig=False)
epochs.plot(scalings=dict(eeg=1.5e-1), n_epochs=5)
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
ica.plot_sources(inst=epochs)
#%% Apply correction
#We have to visually inspect the eog component
eog_component = iex.getBadICAs(patient_number)
ica.exclude = eog_component
epochs_corrected = epochs.copy()
ica.apply(epochs_corrected)
epochs_corrected.plot(scalings=dict(eeg=1.5e-1), n_epochs=5)
#%% Removing epochs with points > 3*std

#We need to know at which epochs the patient loses consciousness and wakes up:
passed_LOC = False
for i in range(epochs_corrected.events.shape[0]):
    if epochs_corrected.events[i][2] == 2 and passed_LOC == False: 
        LOC_epoch = i; passed_LOC = True;
    if epochs_corrected.events[i][2] == 1 and passed_LOC == True:
        ROC_epoch = i; break;

std_channels = [] # (std_values, 3) ,
for i in range(31): 
    std_channels.append([  np.std(epochs_corrected.get_data()[:LOC_epoch,i,:]),          # std pre LOC
                           np.std(epochs_corrected.get_data()[LOC_epoch:ROC_epoch,i,:]), # std asleep
                           np.std(epochs_corrected.get_data()[ROC_epoch:,i,:])           # std post ROC
                        ])                     
epoch_count = 1
bad_epochs_count = 0
state = 0
bad_epochs = []
for epoch in epochs_corrected:
    if epoch_count < LOC_epoch : state = 0
    elif epoch_count > LOC_epoch and epoch_count < ROC_epoch : state = 1
    elif epoch_count > ROC_epoch : state = 2
    for i in range(31):
        if np.max(epoch[i,:]) - np.min(epoch[i,:]) > 7*std_channels[i][state]:# or abs(np.min(epoch[i,:])) > 5*std_channels[i]:
            #print(epoch_count, ';', i);
            bad_epochs_count = bad_epochs_count + 1
            bad_epochs.append(epoch_count)
            break
    epoch_count = epoch_count+1
    #if count == 10:break;
print(bad_epochs_count)       
            
        