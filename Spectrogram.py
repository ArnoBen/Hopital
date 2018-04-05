import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sg
import math

EEGdatacsv = pd.read_csv(r"C:\Users\Arno\OneDrive\Documents\Stage Hopital\Data\Patient112\AS3ExportDataEEG1converted.csv",sep=';')
EEGdata= np.asarray(EEGdatacsv)
#%%
f, t, Sxx = sg.spectrogram(EEGdata[:,1],100)
for i in range(Sxx.shape[0]):
    for j in range(Sxx.shape[1]):    
        SxxReal[i,j]= np.asarray(Sxx.item(i,j).real)
#%%
plt.pcolormesh(t, f, SxxReal)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
#%% 

data=np.asarray(EEGdata)
x=data[:,1]
fig = plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(111)

ax1.set_title("EEG")    
ax1.set_xlabel('time')

ax1.plot(x, c='r', label='the data')

leg = ax1.legend()

plt.show()
#%% 
