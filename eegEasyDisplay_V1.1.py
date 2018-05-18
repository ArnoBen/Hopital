import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from matplotlib.widgets import Button
from scipy import signal as sg

filepath = r"C:\Users\Arno\Downloads\eeg.txt"
sig = pd.read_csv(filepath,sep=';',decimal='.')

#Si 500Hz, downsample 250Hz
if 0.95 <= sig.iloc[500,0] - sig.iloc[0,0] <= 1.05 :
    sig = sig.iloc[::2,:]
#%%
sample_rate = 250
#filtrer
fn=sample_rate/2 #fn: Nyquist frequency = sample_rate/2
b,a=sg.butter(5,(0.3/fn, 50/fn),'bandpass')
#EEGfiltered = sg.filtfilt(b,a,sig['00 E1'])

S, F, T, Img = [],[],[],[]
fields = sig.columns.values.tolist()
window = 2048 #Nombre de points de la fenetre
for i in range (4, 36):
    Stemp, Ftemp, Ttemp, Imgtemp = plt.specgram(sg.filtfilt(b,a,sig[fields[i]]),NFFT=window,Fs=sample_rate,noverlap=window/2);#,vmin=-70,vmax=20);
    # Reduction de la taille des donnees : recuperer uniquements les points < 50Hz
    # Le tableau est de la taille 1025 lignes x n colonnes (varie selon le temps et la taille des fenetres)
    # 125Hz (SamplingRate/2) --> 1025 lignes, donc 50Hz --> 410 lignes.
    Imgtemp = Imgtemp._A[615:1025]  #On prend les 410 dernieres lignes.
    
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
fig = plt.figure(figsize=(20,10))
def plot8spec(index):
    for j in range (1,9):
        subplot(2,4,j)
        current_field = j-1 + (index)*8;
        [Tmax, Tlabel] = axis_parameters(F[j],T[j])
        plt.imshow(Img[current_field],vmin=-70,vmax=-20, aspect='auto', interpolation='none', cmap='viridis', extent = (0,Tmax,0,50))
        spect_dir_path = r"C:\Users\Arno\Documents\Patients\Clement\\"
        spect_img_path = spect_dir_path + str(current_field) + '.png'
        pathlib.Path(spect_dir_path).mkdir(parents=True, exist_ok=True) 
        plt.imsave(fname = spect_img_path, arr = Img[current_field],vmin=-70,vmax=-20, cmap='viridis')
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

## Fleches pour changer de set de canaux
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
#####
for i in range(4):
    plot8spec(i)
