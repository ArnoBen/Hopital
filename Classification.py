import nilearn
import sklearn
import scipy
from scipy.signal import welch
import mne
import numpy as np
import ImportExcel as iex
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from mpl_toolkits.mplot3d import Axes3D

patient_number = 217
sample_rate = 250
epoch_duration = 2 #in seconds
NFFT = sample_rate * epoch_duration
filepath = r'C:\Users\Arno\Documents\Patients\Epochs\patient' + str(patient_number) + '-epo-30Hz.fif'
patient1_mne = mne.read_epochs(filepath)
patient1_np = patient1_mne.get_data()
dataset_size = patient1_np.shape[0]
patient1_mne.ch_names
bad_channels = iex.getBadChannels(patient_number)
#%% Separation training/testing sets
training, testing = np.empty,np.empty
i=0
while i < dataset_size:
    for j in range(4): #4 valeurs sur 5 sont attribuées au training set
        training.append(patient1_np[i+j,:,:])
    testing.append(patient1_np[i+4,:,:])
    if i+5 <= dataset_size: i+=5
    else : break;

#%% Features : standard deviation, power spectrum alpha amplitude

#Standard deviation
std_epochs = [] # (31 channels * n_epochs)
for ch in range(31) : std_epochs.append([]) 
for ch in range(31) :
    if ch in bad_channels : continue
    for epoch in range(dataset_size):
        std_epochs[ch].append(np.std(patient1_np[epoch,ch,:]))

#FFT
fft_alpha_epochs = [] # (31 channels * n_epochs)
fft_beta_epochs = []
for ch in range(31) : fft_alpha_epochs.append([]), fft_beta_epochs.append([])
for ch in range(31) : 
    if ch in bad_channels : continue
    for epoch in range(dataset_size):
        epoch_fft = np.fft.fft(patient1_np[epoch,ch,:])
        mean_alpha_power = np.average(epoch_fft[8:12])
        mean_beta_power = np.average(epoch_fft[20:30])
        fft_alpha_epochs[ch].append(np.abs(mean_alpha_power))
        fft_beta_epochs[ch].append(np.abs(mean_beta_power))

#Power Spectrum Density
psd_alpha_epochs = [] # (31 channels * n_epochs)
psd_beta_epochs = []
for ch in range(31) : psd_alpha_epochs.append([]), psd_beta_epochs.append([])
for ch in range(31) : 
    if ch in bad_channels : continue
    for epoch in range(dataset_size):
        f,Pxx = welch(patient1_np[epoch,ch,:], nfft=NFFT)
        mean_alpha_psd = np.average(Pxx[8:12])
        mean_beta_psd = np.average(Pxx[18:30])
        psd_alpha_epochs[ch].append(np.abs(mean_alpha_psd))
        psd_beta_epochs[ch].append(np.abs(mean_beta_psd))
#%%
plt.figure()
f, Pxx = welch(patient1_np[150,0,:], nfft = NFFT)
subplot(2,1,1)
plt.plot(Pxx[:50])
f, Pxx = welch(patient1_np[1500,0,:], nfft = NFFT)
subplot(2,1,2)
plt.plot(Pxx[:50])
#%%
#Repartition awake/asleep
awake_std = []
awake_fft_alpha = []
awake_fft_beta = []
awake_psd_alpha = []
awake_psd_beta = []

asleep_std = []
asleep_fft_alpha = []
asleep_fft_beta = []
asleep_psd_alpha = []
asleep_psd_beta = []


for ch in range(31) : 
    (awake_std.append([]), awake_fft_alpha.append([]), awake_fft_beta.append([]), awake_psd_alpha.append([]) , awake_psd_beta.append([]),
    asleep_std.append([]), asleep_fft_alpha.append([]),asleep_fft_beta.append([]),asleep_psd_alpha.append([]), asleep_psd_beta.append([]))
events = patient1_mne.events
for ch in range(31):
    if ch in bad_channels : continue
    for epoch in range(dataset_size):
        if events[epoch][2] == 2:
            asleep_std[ch].append(std_epochs[ch][epoch])
            asleep_fft_alpha[ch].append(fft_alpha_epochs[ch][epoch])
            asleep_fft_beta[ch].append(fft_beta_epochs[ch][epoch])
            asleep_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
            asleep_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
        else:
            awake_std[ch].append(std_epochs[ch][epoch])
            awake_fft_alpha[ch].append(fft_alpha_epochs[ch][epoch])
            awake_fft_beta[ch].append(fft_beta_epochs[ch][epoch])
            awake_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
            awake_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
plt.close('all')
for ch in range(31):
    plt.figure()   
    plt.scatter(asleep_std[ch], asleep_psd_beta[ch], marker = '.', s = 2)
    plt.scatter(awake_std[ch], awake_psd_beta[ch], marker = '.', s= 2)

#%%

for ch in range(31):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n = 100
    
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    xs = asleep_std[ch]
    ys = asleep_psd_alpha[ch]
    zs = asleep_psd_beta[ch]
    ax.scatter(xs, ys, zs, c='o', marker='.', alpha = 0.5)
    xs = awake_std[ch]
    ys = awake_psd_alpha[ch]
    zs = awake_psd_beta[ch]
    ax.scatter(xs, ys, zs, c='cyan', marker = '.', aplha = 0.5)
        
    ax.set_xlabel('std')
    ax.set_ylabel('alpha power')
    ax.set_zlabel('beta power')
    
    plt.show()
