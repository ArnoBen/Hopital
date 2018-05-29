import nilearn
import sklearn
import sklearn.metrics as metrics
import scipy
import pylab
import random
from scipy.signal import welch
import mne
import numpy as np
import ImportExcel as iex
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import subplot
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D

sample_rate = 250
epoch_duration = 2 #in seconds
#NFFT = sample_rate * epoch_duration ### Je crois que je me suis gouré
X_patient, y_patient = [], [] #Cette variable contiendra tous les patients séparés

for patient in range(8):
    X_patient.append([]), y_patient.append([])
    for ch in range(31):
        X_patient[patient].append([]), y_patient[patient].append([])

X_all_patients, y_all_patients = [], [] #Cette variable contiendra tous les patients concaténés
X_all_patients_awake, y_all_patients_awake = [], [] #Tous les patients concaténés réveillés
X_all_patients_asleep, y_all_patients_asleep = [], [] #Tous les patients concaténés réveillés endormis

patient_feat_lenghts = []

empty_array = np.array([[],[],[],[],[],[]]).T

for ch in range(31) : 
    X_all_patients.append([])
    X_all_patients_awake.append([])
    X_all_patients_asleep.append([])
    X_all_patients[ch] = empty_array.copy()
    X_all_patients_awake[ch] = empty_array.copy()
    X_all_patients_asleep[ch] = empty_array.copy()

    y_all_patients.append(np.array([]))
    y_all_patients_awake.append(np.array([]))
    y_all_patients_asleep.append(np.array([]))
    
    patient_feat_lenghts.append([])
    
patients = [205,216,217,219,222,224,228,229]
patient_count = 0
for patient_number in patients:
    filepath = r'C:\Users\Arno\Documents\Patients\Epochs\patient' + str(patient_number) + '-epo-30Hz.fif'
    patient_mne = mne.read_epochs(filepath)
    patient_np = patient_mne.get_data()
    dataset_size = patient_np.shape[0]
    patient_mne.ch_names
    bad_channels = iex.getBadChannels(patient_number)
    
    #Features : standard deviation, power spectrum delta/alpha/beta amplitude
    
    #Standard deviation
    std_epochs = [] # (31 channels * n_epochs)
    for ch in range(31) : std_epochs.append([]) 
    for ch in range(31) :
        if ch in bad_channels : continue
        for epoch in range(dataset_size):
            std_epochs[ch].append(np.std(patient_np[epoch,ch,:]))
    
    #Power Spectrum Density
    psd_delta_epochs = []
    psd_theta_epochs = []
    psd_alpha_epochs = [] # (31 channels * n_epochs)
    psd_beta_epochs = []
    psd_epochs = []
    for ch in range(31) : psd_delta_epochs.append([]), psd_theta_epochs.append([]), psd_alpha_epochs.append([]), psd_beta_epochs.append([]), psd_epochs.append([])
    for ch in range(31) : 
        if ch in bad_channels : continue
        for epoch in range(dataset_size):
            f,Pxx = welch(patient_np[epoch,ch,:], nperseg=250, nfft=sample_rate)
            mean_delta_psd = np.average(Pxx[0:4])
            mean_theta_psd = np.average(Pxx[4:8])
            mean_alpha_psd = np.average(Pxx[8:12])
            mean_beta_psd = np.average(Pxx[18:30])
            mean_psd = np.average(Pxx)
            
            psd_delta_epochs[ch].append(np.abs(mean_delta_psd))
            psd_theta_epochs[ch].append(np.abs(mean_theta_psd))
            psd_alpha_epochs[ch].append(np.abs(mean_alpha_psd))
            psd_beta_epochs[ch].append(np.abs(mean_beta_psd))
            psd_epochs[ch].append(np.abs(mean_psd))
    
    
    #Repartition awake/asleep
    awake_std = []
    awake_psd_delta = []
    awake_psd_theta = []
    awake_psd_alpha = []
    awake_psd_beta = []
    awake_psd = []
    
    asleep_std = []
    asleep_psd_delta = []
    asleep_psd_theta = []
    asleep_psd_alpha = []
    asleep_psd_beta = []
    asleep_psd = []
    
    for ch in range(31) : 
        (awake_std.append([]), awake_psd_alpha.append([]) , awake_psd_beta.append([]),
        asleep_std.append([]),asleep_psd_alpha.append([]), asleep_psd_beta.append([]),
        awake_psd_delta.append([]), asleep_psd_delta.append([]), awake_psd_theta.append([]),
        asleep_psd_theta.append([]), awake_psd.append([]), asleep_psd.append([]))
        
    events = patient_mne.events
    for ch in range(31):
        if ch in bad_channels : continue
        for epoch in range(dataset_size):
            if events[epoch][2] == 2:
                asleep_std[ch].append(std_epochs[ch][epoch])
                
                asleep_psd_delta[ch].append(psd_delta_epochs[ch][epoch])
                asleep_psd_theta[ch].append(psd_theta_epochs[ch][epoch])
                asleep_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
                asleep_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
                asleep_psd[ch].append(psd_epochs[ch][epoch])
            else:
                awake_std[ch].append(std_epochs[ch][epoch])

                awake_psd_delta[ch].append(psd_delta_epochs[ch][epoch])
                awake_psd_theta[ch].append(psd_theta_epochs[ch][epoch])
                awake_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
                awake_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
                awake_psd[ch].append(psd_epochs[ch][epoch])
    # repartition training testing
    scores = []
    for ch in range (31):
        #if ch in bad_channels : X = np.zeros([patient_np.shape[0],3]); y = np.zeros([patient_np.shape[0],3]) 
        #On veut un tableau de données (n_samples x n_features), ici n points et 3 features
        std, psd_delta, psd_theta, psd_alpha, psd_beta, psd_mean = [],[],[],[],[],[]
        std = asleep_std[ch].copy() ;              std.extend(awake_std[ch])
        psd_delta = asleep_psd_delta[ch].copy() ;  psd_delta.extend(awake_psd_delta[ch])
        psd_theta = asleep_psd_theta[ch].copy() ;  psd_theta.extend(awake_psd_theta[ch])
        psd_alpha = asleep_psd_alpha[ch].copy() ;  psd_alpha.extend(awake_psd_alpha[ch])
        psd_beta = asleep_psd_beta[ch].copy()   ;  psd_beta.extend(awake_psd_beta[ch])
        psd_mean = asleep_psd[ch].copy()        ;  psd_mean.extend(awake_psd[ch])
        X = np.array([np.array(std), np.array(psd_delta), np.array(psd_theta), np.array(psd_alpha), np.array(psd_beta), np.array(psd_mean)])
        X = X.T
        y = np.array([])
        for i in range(len(asleep_std[ch])) : y = np.append(y, 2)
        for i in range(len(awake_std[ch])) : y = np.append(y, 1)
        
        X_patient[patient_count][ch] = X
        y_patient[patient_count][ch] = y
        
        X_all_patients[ch] = np.concatenate((X_all_patients[ch], X))
        y_all_patients[ch] = np.concatenate((y_all_patients[ch], y))
        patient_feat_lenghts[ch].append(X.shape[0])
    patient_count+=1
    #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_all_patients,y_all_patients)

from sklearn.neighbors import KNeighborsClassifier
scores = []
for ch in range(31) : 
    #if ch in [5,8,9,11,15,16,17,21,20,22,25] : continue;
    print(ch)
    X_train = X_all_patients[ch][:np.sum(patient_feat_lenghts[ch][:6])]
    y_train = y_all_patients[ch][:np.sum(patient_feat_lenghts[ch][:6])]
    X_test = X_all_patients[ch][np.sum(patient_feat_lenghts[ch][:6]):]
    y_test = y_all_patients[ch][np.sum(patient_feat_lenghts[ch][:6]):]
    Knn = KNeighborsClassifier(n_neighbors=8)
    Knn.fit(X_train,y_train)
    score = Knn.score(X_test, y_test)
    scores.append(score)
plt.scatter(range(1,32),scores)
#%% Cross-validation
#for train_indices, test_indices in k_fold.split(X_all_patients[0]):
#    print('Train: %s | test: %s' % (train_indices, test_indices))
auc_valid, auc_test = np.zeros([14, 10, 31]), np.zeros([14, 10, 31]) # (nb k testés * patients * channels)
#X_train = X_patient[:6]

# Méthode : 
# Créer 2 matrices 3D de scores : la première sur Xvalid, la 2e sur Xtest.
# Une fois que les auc sont enregistrés, on regarde le meilleur k sur les auc de Xvalid,
# et on regarde à quel score il correspond sur Xtest, ce qui sera le score réel.
for k in range(1,15):
    
    knn_cv = KNeighborsClassifier(k)
    for n in range(10):
        
        sample = random.sample(range(8),8)  #Sample de patients
        #Séparation des patients en train, valid et test:
        X_train_patient, y_train_patient, X_valid_patient, y_valid_patient, X_test_patient, y_test_patient = [],[],[],[],[],[]
        for ch in range(31) : #Initialisation
            X_train_patient.append(empty_array), y_train_patient.append([]), 
            X_valid_patient.append(empty_array), y_valid_patient.append([]),
            X_test_patient.append([]),           y_test_patient.append([])      
            for i in range(5) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
                X_train_patient[ch] = np.vstack((X_train_patient[ch], X_patient[sample[i]][ch]))
                y_train_patient[ch] = np.concatenate((y_train_patient[ch], y_patient[sample[i]][ch]))
            for i in range(2) :
                X_valid_patient[ch] = np.vstack((X_valid_patient[ch], X_patient[sample[i+5]][ch]))
                y_valid_patient[ch] = np.concatenate((y_valid_patient[ch], y_patient[sample[i+5]][ch]))
            X_test_patient[ch] = X_patient[sample[7]][ch]
            y_test_patient[ch] = y_patient[sample[7]][ch]
        
        for ch in range(31):
            
            ind = np.random.choice(10,10)
            X_train , y_train = X_train_patient[ch] , y_train_patient[ch]
            X_valid , y_valid = X_valid_patient[ch] , y_valid_patient[ch]
            X_test  , y_test  = X_test_patient[ch]  , y_test_patient[ch]
            
            if X_train.shape[0] * X_valid.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
            
            knn_cv.fit(X_train, y_train)
            prediction_prob_valid = knn_cv.predict_proba(X_valid)
            prediction_prob_test = knn_cv.predict_proba(X_test)
            
            for i in range(len(y_valid_patient[ch])) : # Y a une couille parce qu'il veut que des évènements binaires donc asleep passe de 2 à 0 ; awake 1
                if y_valid_patient[ch][i] == 2 : y_valid_patient[ch][i] = 0
            for i in range(len(y_test_patient[ch])) : # Y a une couille parce qu'il veut que des évènements binaires donc asleep passe de 2 à 0 ; awake 1
                if y_test_patient[ch][i] == 2 : y_test_patient[ch][i] = 0
            auc_valid[k-1,n,ch] = metrics.roc_auc_score(y_valid_patient[ch], prediction_prob_valid[:,1])
            auc_test[k-1,n,ch] = metrics.roc_auc_score(y_test_patient[ch], prediction_prob_test[:,1])
    print(k)
auc_valid_mean = np.mean(auc_valid, axis=1)
auc_test_mean = np.mean(auc_test, axis=1)    
pylab.unravel_index(auc_valid_mean.argmax(), auc_valid_mean.shape)
#        auc_valid_n.append(auc_)
#    auc_cv_k[k] = np.mean(auc_cv_n)
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0)),
#    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    MLPClassifier(alpha=1),
#    AdaBoostClassifier(),
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis()]
scores = []
#for classifier in classifiers:
#    classifier.fit(X_train, y_train)
#    score = classifier.score(X_test, y_test)
#    scores.append(score)
for k in range(2,20):
    Knn = KNeighborsClassifier(n_neighbors=k)
    Knn.fit(X_train,y_train)
    score = Knn5.score(X_test, y_test)
    scores.append(score)
print(scores)
#%% 3D plots
X_backup = X_all_patients.copy()
y_backup = y_all_patients.copy()
for ch in range (31):
    X_all_patients_asleep[ch] = np.vstack((X_all_patients_asleep[ch], X_all_patients[ch][y_all_patients[ch] == 2]))
    X_all_patients_awake[ch] = np.vstack((X_all_patients_awake[ch], X_all_patients[ch][y_all_patients[ch] == 1]))
#        if y_all_patients[ch][i] == 2:
#            X_all_patients_asleep[ch] = np.concatenate(X_all_patients_asleep[ch], X_all_patients[ch][i])
#        elif y_all_patients[ch][i] == 1:
#            X_all_patients_awake[ch] = np.concatenate(X_all_patients_awake[ch], X_all_patients[ch][i])

for ch in range(25,27):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n = 100
    
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    xs = X_all_patients_asleep[ch][::20,0]
    ys = X_all_patients_asleep[ch][::20,1]
    zs = X_all_patients_asleep[ch][::20,2]
    ax.scatter(xs, ys, zs, c='b', marker='.', label = 'asleep')
    xs = X_all_patients_awake[ch][::20,0]
    ys = X_all_patients_awake[ch][::20,1]
    zs = X_all_patients_awake[ch][::20,2]
    ax.scatter(xs, ys, zs, c='orange', marker = '.', label = 'awake')
    ax.legend()    
    ax.set_xlabel('std')
    ax.set_ylabel('delta power')
    ax.set_zlabel('alpha power')
    
    plt.show()
#%% 2D plots
    
for ch in range(5):
    plt.figure()   
    plt.scatter(X_all_patients_asleep[ch][::20,0], X_all_patients_asleep[ch][::20,1], marker = '.', s = 4, color = 'b', alpha = 0.4, label = 'asleep')
    plt.scatter(X_all_patients_awake[ch][::20,0], X_all_patients_awake[ch][::20,1], marker = '.', s= 4, color = 'orange', alpha = 0.4, label = 'awake')
    plt.legend()
#%%
epoch_nbr = 50
channel_nbr = 10
faulty_epoch = patient_np[epoch_nbr,channel_nbr,:]

plt.figure()
plt.suptitle(str(events[epoch_nbr][2]))
plt.subplot(2,1,1)
plt.plot(faulty_epoch)

f, faulty_epoch_psd = Pxx = welch(faulty_epoch, nfft=epoch_duration * sample_rate)
plt.subplot(2,1,2)

plt.plot(faulty_epoch_psd[:30])
