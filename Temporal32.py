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

#%%
from matplotlib.ticker import FuncFormatter, MaxNLocator

def axis_parameters(T):
    Tlabel = 'Time (secs)'
    Tmax = T[len(EEGdatacsv[fields[0]])-1] #Dernière valeur du field 0 donc temps total en secondes
    if Tmax> 60*60*3:
       Tmax = Tmax/(60*60)
       Tlabel = 'Time (hours)'
    elif Tmax > 60*3:
       Tmax = Tmax/60
       Tlabel = 'Time (mins)'
    return [Tmax, Tlabel ] 

labels=[]
if   axis_parameters(EEGdatacsv[fields[0]])[1] == 'Time (mins)':  labels = EEGdatacsv[fields[0]]/60
elif axis_parameters(EEGdatacsv[fields[0]])[1] == 'Time (hours)':  labels = (EEGdatacsv[fields[0]]/(60*60))
else: labels = EEGdatacsv[fields[0]]

def format_fn(tick_val, tick_pos): 
    if int(tick_val) in EEGdatacsv[fields[0]]:
        return int(labels[int(tick_val)])
    else:
        return ''

electrode_sets = ['1-8', '9-16', '17-23', '24-32']
plt.close('all')
fig = plt.figure(figsize=(20,10))
ax = []
def plot8temp(index):
    apply_filter = False
    
    for k in range(len(ax)): fig.delaxes(ax[k])
    ax.clear()
    for j in range (1,9):
        sb = subplot(2,4,j)
        ax.append(sb)
        [Tmax, Tlabel] = axis_parameters(EEGdatacsv[fields[0]])
        current_field = j-1 + index*8;     
        if apply_filter==False: plt.plot(EEGdatacsv[fields[current_field+4]])
        else :                  plt.plot(sg.filtfilt(b,a,EEGdatacsv[fields[current_field+4]]))
        sb.xaxis.set_major_formatter(FuncFormatter(format_fn))
        sb.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(fields[current_field+4])
        plt.suptitle(electrode_sets[index], fontsize=24, x=0.514, y=0.962)
        if j>=5: plt.xlabel(Tlabel)
        plt.draw()
plot8temp(0) 
    
class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind>3: self.ind=0
        plot8temp(self.ind)

    def prev(self, event):
        self.ind -= 1
        if self.ind<0: self.ind=3            
        plot8temp(self.ind)
        
callback = Index()
axprev = plt.axes([0.4, 0.92, 0.08, 0.05]) #x0, x1, width, height
axnext = plt.axes([0.54, 0.92, 0.08, 0.05])
#axindex = plt.axes([0.49, 0.92, 0.08, 0.05])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()
                