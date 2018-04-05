import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplot
from matplotlib.widgets import Button
from scipy import signal as sg
import math

EEGdatacsv = pd.read_csv(r"C:\Users\Arno\Documents\Patients\Arno\eeg.txt",sep=';',decimal='.')

#%%
#Si 500Hz, downsample 250Hz
if 0.95 <= EEGdatacsv.iloc[500,0] - EEGdatacsv.iloc[0,0] <= 1.05 : #Si le point 500 est à quasiment 1 seconde
    EEGdatacsv = EEGdatacsv.iloc[::2,:]
#%%
sample_rate = 250
#filtrer
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(0.3/fn, 50/fn),'bandpass')
#EEGfiltered = sg.filtfilt(b,a,EEGdatacsv['00 E1'])

fields = EEGdatacsv.columns.values.tolist()

#%% Without filter
plt.close('all')
for i in range(4):
    plt.figure()
    for j in range (1,9):
        sb = subplot(2,4,j)
        current_field = j-1 + i*8;        
        plt.plot(EEGdatacsv[fields[current_field+4]])
        
        plt.title(fields[current_field+4])
        
        
#%% With filter
plt.close('all')
for i in range(4):
    plt.figure()
    for j in range (1,9):
        sb = subplot(2,4,j)
        current_field = j-1 + i*8;        
        plt.plot(sg.filtfilt(b,a,EEGdatacsv[fields[current_field+4]]))
        
        plt.title(fields[current_field+4])
                