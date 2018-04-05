import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat4py as mp
import scipy as sc
import signal as sg
if not 'EEGdatacsv' in locals():
    EEGdatacsv = pd.read_csv(r"C:\Users\Arno\OneDrive\Documents\Stage Hopital\Data\2018-02-16_09.04.05\eeg.txt",sep=';',decimal='.')

#%% Reduce size, 500Hz --> 100Hz
#   Test = EEGdatacsv.iloc[::5,:]; #Récupère toute la table en prenant 1 élément sur 5. 
#%% Export to matlab
EEGdataReduced = EEGdatacsv.iloc[1200000:1700000,:]
EEGdata = EEGdataReduced.apply(tuple).to_dict()
print('yay')
mp.savemat('Surgery_500Hz.mat',{'EEG': EEGdata})
#%% afficher spectogram
S, F, T, Img = plt.specgram(EEGdataReduced['00 E1'],NFFT=1000,Fs=500,noverlap=1024/2,vmin=-70,vmax=-20)
plt.colorbar()
plt.gca().set_ylim([0,50])
#%% filtrer
fn=500/2
b,a=sg.butter(5,(0.3/fn, 50/fn),'bandpass')
EEGfiltered = sg.filtfilt(b,a,EEGdatacsv['00 E1'])
plt.specgram(EEGfiltered,NFFT=500,Fs=500,noverlap=500/2,vmin=-80,vmax=-50)

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
