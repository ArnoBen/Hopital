import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplot
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
import ImportExcel as iex
from scipy import signal as sg
import math
import os
import pathlib
patient_number = 214
filepath = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\eeg.txt"
EEGdatacsv = pd.read_csv(filepath,sep=';',decimal='.')

#Si 500Hz, downsample 250Hz
if 0.95 <= EEGdatacsv.iloc[500,0] - EEGdatacsv.iloc[0,0] <= 1.05 : #Si le point 500 est à quasiment 1 seconde
    EEGdatacsv = EEGdatacsv.iloc[::2,:]
#%%
sample_rate = 250
#filtrer
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(0.3/fn, 50/fn),'bandpass')
#EEGfiltered = sg.filtfilt(b,a,EEGdatacsv['00 E1'])

S, F, T, Img = [],[],[],[]
fields = EEGdatacsv.columns.values.tolist()
window = 2048 #Nombre de points de la fenêtre
for i in range (4, 36):
    Stemp, Ftemp, Ttemp, Imgtemp = plt.specgram(sg.filtfilt(b,a,EEGdatacsv[fields[i]]),NFFT=window,Fs=sample_rate,noverlap=window/2);#,vmin=-70,vmax=20);
    # Reduction de la taille des données : récupérer uniquements les points < 50Hz
    # Le tableau est de la taille 1025 lignes x n colonnes (varie selon le temps et la taille des fenêtres)
    # 125Hz (SamplingRate/2) --> 1025 lignes, donc 50Hz --> 410 lignes.
    Imgtemp = Imgtemp._A[615:1025]  #Le tableau est inversé donc on prend les 410 dernières lignes.
    
    S.append(Stemp)
    F.append(Ftemp)
    T.append(Ttemp)
    Img.append(Imgtemp)
    print(i)
plt.close()
#%% 

##########

def axis_parameters(F,T):
    Tlabel = 'Time (secs)'
    Tmax = T.max()
    if T.max() > 60*60*3:
       Tmax = T.max()/(60*60)
       Tlabel = 'Time (hours)'
    elif T.max() > 60*3:
       Tmax = T.max()/60
       Tlabel = 'Time (mins)'
    return [Tmax, Tlabel ] 

electrode_sets = ['1-8', '9-16', '17-23', '24-32']
plt.close('all')
fig = plt.figure(figsize=(20,10), num='Patient ' + str(patient_number))
ax = []
axvlines = np.empty((4,0)).tolist()
def plot8spec(index):
    for k in range(len(ax)): fig.delaxes(ax[k])
    ax.clear()
    for axvline in axvlines : axvline.clear()
    for j in range (1,9):
        sb = subplot(2,4,j)
        ax.append(sb)
        current_field = j-1 + (index)*8;
        [Tmax, Tlabel] = axis_parameters(F[j],T[j])
        plt.imshow(Img[current_field],vmin=-70,vmax=-20, aspect='auto', interpolation='none', cmap='viridis', extent = (0,Tmax,0,50))
        spect_dir_path = r"C:\Users\Arno\Documents\Patients\\" + str(patient_number) + "\Spectrograms\\"
        spect_img_path = spect_dir_path + str(current_field) + ' - ' + str(fields[(current_field)+4]) + '.png'
        pathlib.Path(spect_dir_path).mkdir(parents=True, exist_ok=True) 
        #Avant de créer l'image, on vérifie si elle existe déja
        if not os.path.exists(spect_img_path): plt.imsave(fname = spect_img_path, arr = Img[current_field],vmin=-70,vmax=-20, cmap='viridis')
        #plt.cm('plasma')
        
        ## Axes verticaux pour marquer les temps
        axvlines[0].append(plt.axvline(x=iex.getPropofolTime(patient_number, timestyle=Tlabel), color='w', linestyle = '--'))
        axvlines[1].append(plt.axvline(x=iex.getSedationStopTime(patient_number, timestyle=Tlabel), color='lightgrey', linestyle = '--'))    
        axvlines[2].append(plt.axvline(x=iex.getSutureEndTime(patient_number, timestyle=Tlabel), color='rosybrown', linestyle = '--'))    
        axvlines[3].append(plt.axvline(x=iex.getWakeUpTime(patient_number, timestyle=Tlabel), color='r', linestyle = '--'))
        for i in range(4) : axvlines[i][j-1].set_visible(check.get_status()[i])
        
        plt.ylim([0,50]) 
        plt.title(fields[(current_field)+4])
        if (j==1 or j==5):plt.ylabel('Frequency (Hz)')
        if j>=5: plt.xlabel(Tlabel)
    cbaxes = plt.axes([0.92, 0.112, 0.02, 0.77]) #x0, x1, width, height
    plt.colorbar(cax = cbaxes)
    print(index)
    plt.suptitle(electrode_sets[index], fontsize=24, x=0.514, y=0.962)
    plt.draw()
    plt.show()

##### Buttons #####

## Flèches pour changer de set de canaux
class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind>3: self.ind=0
        plot8spec(self.ind)

    def prev(self, event):
        self.ind -= 1
        if self.ind<0: self.ind=3            
        plot8spec(self.ind)
        
callback = Index()
axprev = plt.axes([0.4, 0.92, 0.08, 0.05]) #x0, x1, width, height
axnext = plt.axes([0.54, 0.92, 0.08, 0.05])
#axindex = plt.axes([0.49, 0.92, 0.08, 0.05])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()

## Checkboxes pour afficher les axes
def displayAxvlines(label):
    if label == 'Propofol'      : index_axv = 0
    if label == 'Arret Sedation': index_axv = 1
    if label == 'Fermeture'     : index_axv = 2
    if label == 'Reveil'        : index_axv = 3
    for axv in axvlines[index_axv]: axv.set_visible(check.get_status()[index_axv])
    plt.draw()
    plt.show()
    
checkboxax = plt.axes([0.015, 0.45, 0.075, 0.15])
check = CheckButtons(checkboxax, ('Propofol', 'Arret Sedation', 'Fermeture', 'Reveil'), (True,False, False, True))
check.on_clicked(displayAxvlines)

#####

plot8spec(0)