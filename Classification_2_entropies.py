import pylab
import random
import neurokit as nk
from scipy.signal import welch
from scipy import interp
import mne
import numpy as np
import ImportExcel as iex
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import subplot
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

empty_array = np.array([[],[],[],[],[],[],[],[],[],[],[]]).T #As much as features
features_names = ['std', 'entropy', 'spectral entropy', 'psd mean','psd delta','psd theta','psd alpha','psd beta','ratio delta/beta', 'ratio theta/beta','ratio alpha/beta']
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
    
    #Features : standard deviation, entropy, power spectrum delta/alpha/beta amplitude
    
    #Standard deviation
    std_epochs = [] # (31 channels * n_epochs)
    for ch in range(31) : std_epochs.append([]) 
    for ch in range(31) :
        if ch in bad_channels : continue
        for epoch in range(dataset_size):
            std_epochs[ch].append(np.std(patient_np[epoch,ch,:]))
    #Entropy
    entropy_epochs = []
    spectral_entropy_epochs = []
    for ch in range(31) : entropy_epochs.append([]); spectral_entropy_epochs.Append([]);
    for ch in range(31) :
        if ch in bad_channels : continue
        for epoch in range(dataset_size):    
                complexity = nk.complexity(patient_np[epoch,ch,:], sample_rate,shannon=False, sampen=True, multiscale=False, spectral=True, svd=False, correlation=False, higushi=False, petrosian=False, fisher=False, hurst=False, dfa=False, lyap_r=False, lyap_e=False)
                entropy_epochs[ch].append(complexity['Entropy_Sample'])
                spectral_entropy_epochs[ch].append(complexity['])
        print('entropy',ch)
        
    #Power Spectrum Density
    psd_delta_epochs = []
    psd_theta_epochs = []
    psd_alpha_epochs = [] # (31 channels * n_epochs)
    psd_beta_epochs = []
    psd_epochs = []
    
        #Ratios of PSDs
    ratio_delta_beta_epochs = []
    ratio_theta_beta_epochs = []
    ratio_alpha_beta_epochs = []
    
    for ch in range(31) : 
        psd_delta_epochs.append([]), psd_theta_epochs.append([]), psd_alpha_epochs.append([]), psd_beta_epochs.append([]), psd_epochs.append([])
        ratio_delta_beta_epochs.append([]), ratio_theta_beta_epochs.append([]), ratio_alpha_beta_epochs.append([])
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
            
            #Power Spectrum Density ratios
            ratio_delta_beta_epochs[ch].append( psd_delta_epochs[ch][epoch] / psd_beta_epochs[ch][epoch] )
            ratio_theta_beta_epochs[ch].append( psd_theta_epochs[ch][epoch] / psd_beta_epochs[ch][epoch] )
            ratio_alpha_beta_epochs[ch].append( psd_alpha_epochs[ch][epoch] / psd_beta_epochs[ch][epoch] )
             
    

    #Repartition awake/asleep
    awake_std = []              ;   asleep_std = []
    awake_entropy = []          ;   asleep_entropy = []
    awake_spectral_entropy = [] ;   asleep_spectral_entropy = []
    awake_psd = []              ;   asleep_psd = []
    awake_psd_delta = []        ;   asleep_psd_delta = []
    awake_psd_theta = []        ;   asleep_psd_theta = []
    awake_psd_alpha = []        ;   asleep_psd_alpha = []
    awake_psd_beta = []         ;   asleep_psd_beta = []
    awake_ratio_db = []         ;   asleep_ratio_db = [] #delta/beta
    awake_ratio_tb = []         ;   asleep_ratio_tb = [] #theta/beta
    awake_ratio_ab = []         ;   asleep_ratio_ab = [] #alpha/beta
    
    for ch in range(31) : 
        (awake_std.append([])               ,   asleep_std.append([])       ,
         awake_entropy.append([])           ,   asleep_entropy.append([])   ,
         awake_spectral_entropy.append([])  ,   asleep_spectral_entropy.append([])   ,
         awake_psd.append([])               ,   asleep_psd.append([])       ,
         awake_psd_alpha.append([])         ,   asleep_psd_alpha.append([]) ,
         awake_psd_beta.append([])          ,   asleep_psd_beta.append([])  ,
         awake_psd_delta.append([])         ,   asleep_psd_delta.append([]) ,
         awake_psd_theta.append([])         ,   asleep_psd_theta.append([]) , 
         awake_ratio_db.append([])          ,   asleep_ratio_db.append([])  ,
         awake_ratio_tb.append([])          ,   asleep_ratio_tb.append([])  ,
         awake_ratio_ab.append([])          ,   asleep_ratio_ab.append([])  ,
         )
        
    events = patient_mne.events
    for ch in range(31):
        if ch in bad_channels : continue
        for epoch in range(dataset_size):
            if events[epoch][2] == 2:
                asleep_std[ch].append(std_epochs[ch][epoch])
                asleep_entropy[ch].append(entropy_epochs[ch][epoch])
                asleep_spectral_entropy[ch].append(spectral_entropy_epochs[ch][epoch])
                asleep_psd[ch].append(psd_epochs[ch][epoch])
                asleep_psd_delta[ch].append(psd_delta_epochs[ch][epoch])
                asleep_psd_theta[ch].append(psd_theta_epochs[ch][epoch])
                asleep_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
                asleep_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
                asleep_ratio_db[ch].append(ratio_delta_beta_epochs[ch][epoch])
                asleep_ratio_tb[ch].append(ratio_theta_beta_epochs[ch][epoch])
                asleep_ratio_ab[ch].append(ratio_alpha_beta_epochs[ch][epoch])
            else:
                awake_std[ch].append(std_epochs[ch][epoch])
                awake_entropy[ch].append(entropy_epochs[ch][epoch])
                awake_psd[ch].append(psd_epochs[ch][epoch])
                awake_psd_delta[ch].append(psd_delta_epochs[ch][epoch])
                awake_psd_theta[ch].append(psd_theta_epochs[ch][epoch])
                awake_psd_alpha[ch].append(psd_alpha_epochs[ch][epoch])
                awake_psd_beta[ch].append(psd_beta_epochs[ch][epoch])
                awake_ratio_db[ch].append(ratio_delta_beta_epochs[ch][epoch])
                awake_ratio_tb[ch].append(ratio_theta_beta_epochs[ch][epoch])
                awake_ratio_ab[ch].append(ratio_alpha_beta_epochs[ch][epoch])
    # repartition training testing
    scores = []
    for ch in range (31):
        #On veut un tableau de données (n_epochs x n_features)
        std, entropy, psd_delta, psd_theta, psd_alpha, psd_beta, psd_mean, ratio_db, ratio_tb, ratio_ab = [],[],[],[],[],[],[],[],[],[]
        
        std = asleep_std[ch].copy()             ;   std.extend(awake_std[ch])
        entropy = asleep_entropy[ch].copy()     ;   entropy.extend(awake_entropy[ch])
        psd_mean = asleep_psd[ch].copy()        ;   psd_mean.extend(awake_psd[ch])
        psd_delta = asleep_psd_delta[ch].copy() ;   psd_delta.extend(awake_psd_delta[ch])
        psd_theta = asleep_psd_theta[ch].copy() ;   psd_theta.extend(awake_psd_theta[ch])
        psd_alpha = asleep_psd_alpha[ch].copy() ;   psd_alpha.extend(awake_psd_alpha[ch])
        psd_beta = asleep_psd_beta[ch].copy()   ;   psd_beta.extend(awake_psd_beta[ch])
        ratio_db = asleep_ratio_db[ch].copy()   ;   ratio_db.extend(awake_ratio_db[ch])
        ratio_tb = asleep_ratio_tb[ch].copy()   ;   ratio_tb.extend(awake_ratio_tb[ch])
        ratio_ab = asleep_ratio_ab[ch].copy()   ;   ratio_ab.extend(awake_ratio_ab[ch])
        
        X = np.array([np.array(std), np.array(entropy), np.array(psd_mean), np.array(psd_delta), np.array(psd_theta), np.array(psd_alpha),np.array(psd_beta),np.array(ratio_db),np.array(ratio_tb),np.array(ratio_ab)])
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

#%% Save
#np.save('X_patient.npy', X_patient)
#np.save('y_patient.npy', y_patient)
#%% Load
empty_array = np.array([[],[],[],[],[],[],[],[],[],[]]).T #As much as features
X_patient, y_patient = [None, None]
X_patient = np.load('X_patient.npy', X_patient)
y_patient = np.load('y_patient.npy', y_patient)
y_patient-=1
#%% Cross-validation
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
auc_valid, auc_test = np.zeros([168, 31]), np.zeros([168, 31]) # (patients * channels)

# Méthode : 
# Créer 2 matrices 3D de scores : la première sur Xvalid, la 2e sur Xtest.
# Une fois que les auc sont enregistrés, on regarde le meilleur k sur les auc de Xvalid,
# et on regarde à quel score il correspond sur Xtest, ce qui sera le score réel.
#knn_cv = KNeighborsClassifier(k)

#on génère une liste de sous ensembles (a, (b,c)), (a,b,c)€[0,7]^3, a pour valid et (b,c) pour test. On récupère le training en excluant a,b,c.
def generator():
    rep = []
    for i in range(8):
        for j in range(8):
            if i != j:
                for k in range(j):
                    if k != i:
                        rep += [ (i, (k, j)) ]
                        
    return(rep)
#f = [0,1,4,5,8]
f = [0,1,2,3,4,5,6,7,8,9]
############
qda_cv = QuadraticDiscriminantAnalysis()
gnb_cv = GaussianNB()
best_channels = np.zeros(30)
sets_count = 0
for p in generator(): #On fait 10 essais puis on moyenne les résultats des essais
    #sample = random.sample(range(8),8)  #Sample de patients
    #Séparation des patients en train, valid et test:
    X_train_patient, y_train_patient, X_valid_patient, y_valid_patient, X_test_patient, y_test_patient = [],[],[],[],[],[]
    train_list = [0,1,2,3,4,5,6,7]
    train_list.remove(p[0]) ; train_list.remove(p[1][0]) ; train_list.remove(p[1][1])
    for ch in range(31) : #On fait l'essai sur chaque channel, le but étant d'obtenir le meilleur channel
        X_train_patient.append(empty_array), y_train_patient.append([]), 
        X_valid_patient.append(empty_array), y_valid_patient.append([]),
        X_test_patient.append(empty_array),  y_test_patient.append([])      
        
        X_valid_patient[ch] = X_patient[p[0]][ch]
        y_valid_patient[ch] = y_patient[p[0]][ch]
        for i in range(2) :
            X_test_patient[ch] = np.vstack((X_test_patient[ch], X_patient[p[1][i]][ch]))
            y_test_patient[ch] = np.concatenate((y_test_patient[ch], y_patient[p[1][i]][ch]))
        
        for i in range(5) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient[ch] = np.vstack((X_train_patient[ch], X_patient[train_list[i]][ch]))
            y_train_patient[ch] = np.concatenate((y_train_patient[ch], y_patient[train_list[i]][ch]))
        #Réusmé : 5 training, 1 valid, 2 tests
    for ch in range(31):
          
        X_train , y_train = X_train_patient[ch][:,np.array(f)], y_train_patient[ch]
        X_valid , y_valid = X_valid_patient[ch][:,np.array(f)], y_valid_patient[ch]
        X_test  , y_test  = X_test_patient[ch][:,np.array(f)], y_test_patient[ch]
        
        if X_train.shape[0] * X_valid.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        
        gnb_cv.fit(X_train, y_train)
        prediction_prob_valid = gnb_cv.predict_proba(X_valid)
        prediction_prob_test = gnb_cv.predict_proba(X_test)
        
        auc_valid[sets_count,ch] = roc_auc_score(y_valid_patient[ch], prediction_prob_valid[:,1])
        auc_test[sets_count,ch] = roc_auc_score(y_test_patient[ch], prediction_prob_test[:,1])
    sets_count +=1
    print(sets_count)
auc_valid_mean = np.mean(auc_valid, axis=0)
auc_test_mean = np.mean(auc_test, axis=0)

#Idée : faire tourner l'algo 5 fois et noter les 5 meilleurs couples (k, ch) à chaque fois
temp_auc_valid_mean = auc_valid_mean.copy()
temp_auc_test_mean = auc_test_mean.copy()
best_ch_auc = np.zeros([30,3])
best_auc_test = np.zeros(30)

for j in range(30):
    best_ch_auc[j,0] = temp_auc_valid_mean.argmax() #np.array(pylab.unravel_index(temp_auc_valid_mean.argmax(), temp_auc_valid_mean.shape))
    best_ch_auc[j,1] = temp_auc_valid_mean[temp_auc_valid_mean.argmax()]
    best_ch_auc[j,2] = temp_auc_test_mean[temp_auc_valid_mean.argmax()]
    best_channels[j] = best_ch_auc[j,0]
    best_auc_test[j] = auc_test_mean[int(best_ch_auc[j,0])]
    temp_auc_valid_mean[int(best_ch_auc[j,0])] = 0

#Tour 1 : (7,0) ; (8,2) ; (8,0) ; (11,0) ; (4,0) ; (9,0)  ; (8,30) ; auc_test ~= 0,82 -> 0,88  Le channel 0 semble être un bon candidat, 2 fait une apparition
#Tour 2 : (8,0) ; (5,0) ; (6,6) ; (6,2)  ; (5,2) ; (12,0) ; (8,2)  ; auc_test ~= 0,75 -> 0,89   Le channel 0 est toujours en tête, suivi par 2
#Tour 3 : (13,2); (13,0); (12,0); (12,2) ; (13,6); (10,2) ; (10,0) ; auc_test ~= 0,80 -> 0,87  Channel 0 en tête, channel 2 toujous second
#Tour 4 : (6,0) ; (3,0) ; (9,2) ; (9,6)  ; (6,2) ; (12,0) ; (13,0) ; auc_test = 0,84
#Best result : k=11, ch=2 => auc_test_mean = 0,85
#        auc_valid_n.append(auc_)
#    auc_cv_k[k] = np.mean(auc_cv_n)


#%% ROC curve with varying k and best channel
def generator62(): #6 train - 2 test
    rep = []
    for i in range(8):
        for j in range(8):
            if i != j:
                rep += [(i,j)]
                        
    return(rep)

figure()
mean_fpr = np.linspace(0, 1, 100)

ch = 25
aucs_k = []
stds_k = []
for k in range(1,21):
    knn = KNeighborsClassifier(k)
    dtc = sklearn.tree.DecisionTreeClassifier(max_depth=k)
    tprs = [] #true positive rates
    aucs = [] #areas under curves
    for p in generator62(): #On fait n essais puis on moyenne les résultats des essais
        #Séparation des patients en train et test:
        X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
        y_train_patient, y_test_patient = [],[]
        train_list = [0,1,2,3,4,5,6,7]
        train_list.remove(p[0]) ; train_list.remove(p[1])
        for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient = np.vstack((X_train_patient, X_patient[train_list[i]][ch]))
            y_train_patient = np.concatenate((y_train_patient, y_patient[train_list[i]][ch]))
        for i in range(2) :
            X_test_patient = X_patient[p[i]][ch]
            y_test_patient = y_patient[p[i]][ch]
    
        ind = np.random.choice(10,10)
        X_train , y_train = X_train_patient, y_train_patient
        X_test  , y_test  = X_test_patient , y_test_patient
        
        if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        
        #Entrainement pour une répartition aléatoire train/test sur les n :
        dtc.fit(X_train, y_train)
        predict_prob_test = dtc.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test[:, 1]) #On considère positif qd patient endormi
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (n, roc_auc))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #Plot mean auc
    #if k%5 == 0:
    plt.plot(mean_fpr, mean_tpr,
             label=f'k={k} (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    aucs_k.append(mean_auc)
    stds_k.append(std_auc)
    print(k)
#Plot remaining stuff
#std_tpr = np.std(tprs, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)  
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Mean Receiver Operating Characteristic for each amount of neighbors')
plt.title('Mean Receiver Operating Characteristic for each maximum depth')
#plt.suptitle('Nearest Neighbors', fontsize = 16)
plt.suptitle('Decision Tree Classifier', fontsize = 16)
plt.legend(loc="lower right")
#plt.grid()
plt.show()

figure()
#plt.bar(np.arange(1,k+1,1),aucs_k,0.1)
plt.errorbar(np.arange(1,k+1,1), aucs_k, (np.array(stds_k)/2).tolist(), linestyle = 'None', marker = 'o', capsize = 3, ecolor='r')
plt.xticks(np.arange(1,k+1,1))
plt.ylim([0.70,0.9])
plt.grid(axis = 'x')
plt.xlabel('Maximum Depth')
plt.xlabel('Max depth')
plt.ylabel('Mean AUC')
plt.title('Determination of best maximum depth')
#plt.suptitle('Nearest Neighbors', fontsize = 16)
#plt.suptitle('Decision Tree Classifier', fontsize = 16)
plt.ylim([0.6,0.95])

#%% Multiple model comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Decision Tree", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(6),
    DecisionTreeClassifier(max_depth=3),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
scores = []

tprs = [] #true positive rates
aucs = [] #areas under curves
model_aucs = []
model_stds = []

mean_fpr = np.linspace(0, 1, 100)

ch = 29

for name, clf in zip(names, classifiers):
    figure()
    print(name)
    for n in range(50): #On fait 10 essais puis on moyenne les résultats des essais
        sample = random.sample(range(8),8)  #Sample de patients
        #Séparation des patients en train et test:
        X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
        y_train_patient, y_test_patient = [],[]
       
        for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient = np.vstack((X_train_patient, X_patient[sample[i]][ch]))
            y_train_patient = np.concatenate((y_train_patient, y_patient[sample[i]][ch]))
        for i in range(2) :
            X_test_patient = X_patient[sample[i+6]][ch]
            y_test_patient = y_patient[sample[i+6]][ch]
    
        ind = np.random.choice(10,10)
        X_train , y_train = X_train_patient, y_train_patient
        X_test  , y_test  = X_test_patient , y_test_patient
        
        if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        
        #Entrainement pour une répartition aléatoire train/test sur les n :
        clf.fit(X_train, y_train)
        predict_prob_test = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test[:, 1]) #On considère positif qd patient endormi
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (n, roc_auc))
        print(n)
    #Plot remaining stuff
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    model_aucs.append(mean_auc)
    model_stds.append(std_auc)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plot std in grey :
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()

figure()
#plt.bar(np.arange(1,k+1,1),aucs_k,0.1)
plt.errorbar(np.arange(1,6,1), model_aucs, (np.array(model_stds)/2).tolist(), linestyle = 'None', marker = 'o', capsize = 3, ecolor='r')
plt.xticks(np.arange(1,6,1), names, rotation='vertical')
plt.ylim([0.75,0.9])
plt.grid(axis = 'x')#
plt.xlabel('')
plt.ylabel('Mean AUC for 50 tries')
plt.title('Determination of best model',fontsize = 14)
plt.ylim([0.75,0.95])
#%% ROC curve with best k and best channel
figure()

tprs = [] #true positive rates
aucs = [] #areas under curves
mean_fpr = np.linspace(0, 1, 100)

k = 6
ch = 29
knn = KNeighborsClassifier(k)
gnb = GaussianNB()
#f = [0,3,4,7]
f = [0,1,4,5,8]
predict_probas = []
sets_count = 0
for p in generator62(): #On fait 10 essais puis on moyenne les résultats des essais
    sample = random.sample(range(8),8)  #Sample de patients
    #Séparation des patients en train et test:
    X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
    y_train_patient, y_test_patient = [],[]
    #X_train_patient, y_train_patient, X_valid_patient, y_valid_patient, X_test_patient, y_test_patient = [],[],[],[],[],[]
    train_list = [0,1,2,3,4,5,6,7]
    train_list.remove(p[0]) ; train_list.remove(p[1]) ;
   
    for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
        X_train_patient = np.vstack((X_train_patient, X_patient[train_list[i]][ch]))
        y_train_patient = np.concatenate((y_train_patient, y_patient[train_list[i]][ch]))
    for i in range(2) :
        X_test_patient = X_patient[p[i]][ch]
        y_test_patient = y_patient[p[i]][ch]
    
    X_train , y_train = X_train_patient[:,np.array(f)], y_train_patient
    X_test  , y_test  = X_test_patient[:,np.array(f)], y_test_patient
    
    if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
    #Entrainement pour une répartition aléatoire train/test sur les n :
    gnb.fit(X_train, y_train)
    predict_prob_test = gnb.predict_proba(X_test) ; predict_probas.append(predict_prob_test)
    fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test[:, 1]) #On considère positif qd patient endormi
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    if sets_count %5 == 0 : plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Try n°%d : AUC = %0.2f' % (sets_count, roc_auc))
    sets_count +=1
    print(sets_count)
#Plot remaining stuff
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plot std in grey :
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Naive Bayes', fontsize = 16)
plt.legend(loc="lower right")
plt.show()
#%% Features selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def generator62(): #6 train - 2 test
    rep = []
    for i in range(8):
        for j in range(8):
            if i != j:
                rep += [(i,j)]
                        
    return(rep)
    
names = ["Nearest Neighbors", "Decision Tree", "AdaBoost",
         "Naive Bayes", "QDA"]
from itertools import combinations, chain
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
figure()
k = 20
ch = 29
knn = KNeighborsClassifier(k)
qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()
dtc = DecisionTreeClassifier(max_depth = 4)

feature_sets = []
mean_aucs = []
std_aucs = []
feature_combinations = powerset(np.arange(0,10,1))
nb_feature_combinations = 0
for f in feature_combinations:
    if f == () : continue
    #if len(np.array(f)) < 3 or len(np.array(f))>5: continue
    tprs = [] #true positive rates
    aucs = [] #areas under curves
    #Feature selection
    
    #if f==0 : nb_features = 9 #Au premier essai on essaye avec toutes les features
    #else: nb_features = random.randint(4,8)
    
    #features = np.zeros(nb_features)
    #feature_sample = random.sample(range(9),9)  #Sample de features
    #for i in range(nb_features) : features[i] = feature_sample[i]
    feature_sets.append(f)
    for p in generator62(): #On fait n essais puis on moyenne les résultats des essais
        sample = random.sample(range(8),8)  #Sample de patients
        #Séparation des patients en train et test:
        X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
        y_train_patient, y_test_patient = [],[]
        train_list = [0,1,2,3,4,5,6,7]
        train_list.remove(p[0]) ; train_list.remove(p[1])
        for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient = np.vstack((X_train_patient, X_patient[train_list[i]][ch]))
            y_train_patient = np.concatenate((y_train_patient, y_patient[train_list[i]][ch]))
        for i in range(2) :
            X_test_patient = X_patient[p[i]][ch]
            y_test_patient = y_patient[p[i]][ch]
        
#        X_train , y_train = X_train_patient[:,feature_sample[:nb_features]], y_train_patient
#        X_test  , y_test  = X_test_patient[:,feature_sample[:nb_features]], y_test_patient
        X_train , y_train = X_train_patient[:,np.array(f)], y_train_patient
        X_test  , y_test  = X_test_patient[:,np.array(f)], y_test_patient
        if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        #Entrainement pour une répartition aléatoire train/test sur les n :
        dtc.fit(X_train, y_train)
        predict_prob_test = dtc.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test[:, 1]) #On considère positif qd patient endormi
        #tprs.append(interp(mean_fpr, fpr, tpr))
        #tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    std_auc = np.std(aucs)
    mean_aucs.append(np.mean(aucs)) 
    std_aucs.append(std_auc)
    nb_feature_combinations +=1
    print('f:',f)
plt.errorbar(np.arange(0,nb_feature_combinations,1), mean_aucs, (np.array(std_aucs)/2).tolist(), linestyle = 'None', marker = 'o', capsize = 3, ecolor='r')
plt.plot(mean_aucs)
plt.show()
#%% Maximizing Sensibility
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Decision Tree", "AdaBoost",
         "Naive Bayes", "QDA"]
from itertools import combinations, chain
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
figure()
k = 5
ch = 25
knn = KNeighborsClassifier(k)
qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()
dtc = DecisionTreeClassifier(max_depth = 5)

feature_sets = []
mean_aucs,mean_tprs = [], []
std_aucs, std_tprs = [], []

feature_combinations = powerset(np.arange(0,9,1))
nb_feature_combinations = 0
for f in feature_combinations:
    if f == () : continue
    #if len(np.array(f)) < 3 or len(np.array(f))>5: continue
    tprs = [] #true positive rates or Sensibility
    aucs = [] #areas under curves
    #Feature selection
    if f==0 : nb_features = 9 #Au premier essai on essaye avec toutes les features
    else: nb_features = random.randint(4,8)
    
    #features = np.zeros(nb_features)
    #feature_sample = random.sample(range(9),9)  #Sample de features
    #for i in range(nb_features) : features[i] = feature_sample[i]
    feature_sets.append(f)
    for n in range(30): #On fait n essais puis on moyenne les résultats des essais
        sample = random.sample(range(8),8)  #Sample de patients
        #Séparation des patients en train et test:
        X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
        y_train_patient, y_test_patient = [],[]
        for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient = np.vstack((X_train_patient, X_patient[sample[i]][ch]))
            y_train_patient = np.concatenate((y_train_patient, y_patient[sample[i]][ch]))
        for i in range(2) :
            X_test_patient = X_patient[sample[i+6]][ch]
            y_test_patient = y_patient[sample[i+6]][ch]
        ind = np.random.choice(10,10)
        
#        X_train , y_train = X_train_patient[:,feature_sample[:nb_features]], y_train_patient
#        X_test  , y_test  = X_test_patient[:,feature_sample[:nb_features]], y_test_patient
        X_train , y_train = X_train_patient[:,np.array(f)], y_train_patient
        X_test  , y_test  = X_test_patient[:,np.array(f)], y_test_patient
        if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        #Entrainement pour une répartition aléatoire train/test sur les n :
        gnb.fit(X_train, y_train)
        predict_prob_test = gnb.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test[:, 1]) #On considère positif qd patient endormi
        tprs.append(interp(mean_fpr, fpr, tpr))
        #tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #tprs.append(tpr)
    mean_aucs.append(np.mean(aucs)) 
    std_aucs.append(np.std(aucs))
    mean_tprs.append(np.mean(tprs))
    std_tprs.append(np.std(tprs))
    nb_feature_combinations +=1
    print('f:',f)
plt.errorbar(np.arange(0,nb_feature_combinations,1), mean_tprs, (np.array(std_tprs)/2).tolist(), linestyle = 'None', marker = 'o', capsize = 3, ecolor='r')
plt.plot(mean_aucs)
plt.show()
#%%
figure()
best_model_auc = [0.8777002764152415, 0.8750416743454615, 0.9188999891477102, 0.9262685148653317]
std_best_model_auc = [0.050151486399583924,0.040834078188116706,0.042398924884960346,0.040575981781962325]
best_feature_sets = ['0, 1, 2, 3, 4, 6, 9', '1,9', '0, 1, 6, 8', '0, 1, 4, 5, 8']
model_names = ['Nearest Neighbors \n features : 0, 1, 2, 3, 4, 6, 9 ', 'Decision Tree\n features : 1,9', 'QDA\n features : 0, 1, 6, 8', 'Naive Bayes\n features : 0, 1, 4, 5, 8']
plt.errorbar([0,1,2,3], best_model_auc, (np.array(std_best_model_auc)/2).tolist(),linestyle='None', marker = 'o', capsize = 3, ecolor='r')
plt.ylabel('Mean AUC for 50 tries with best set of features', fontsize = 14)
plt.xticks([0,1,2,3],model_names, fontsize = 14)
plt.title('Model Comparison', fontsize = 20)
#%% Confusion Matrix
while True:
    tprs = [] #true positive rates
    aucs = [] #areas under curves
    
    k = 6
    ch = 29
    knn = KNeighborsClassifier(k)
    gnb = GaussianNB()
    tnrs, fprs, fnrs, tprs = [],[],[],[]
    accuracies = []
    precisions = []
    specificities = []
    sensitivities = []
    f = [0,1,4,5,8]
    
    for p in generator62(): #On fait 10 essais puis on moyenne les résultats des essais
        sample = random.sample(range(8),8)  #Sample de patients
        #Séparation des patients en train et test:
        X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
        y_train_patient, y_test_patient = [],[]
        #X_train_patient, y_train_patient, X_valid_patient, y_valid_patient, X_test_patient, y_test_patient = [],[],[],[],[],[]
        train_list = [0,1,2,3,4,5,6,7]
        train_list.remove(p[0]) ; train_list.remove(p[1]) ;
       
        for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
            X_train_patient = np.vstack((X_train_patient, X_patient[train_list[i]][ch]))
            y_train_patient = np.concatenate((y_train_patient, y_patient[train_list[i]][ch]))
        for i in range(2) :
            X_test_patient = X_patient[p[i]][ch]
            y_test_patient = y_patient[p[i]][ch]
        
        X_train , y_train = X_train_patient[:,np.array(f)], y_train_patient
        X_test  , y_test  = X_test_patient[:,np.array(f)], y_test_patient
    
        
        #if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
        
        #Entrainement pour une répartition aléatoire train/test sur les n :
        gnb.fit(X_train, y_train)
        predict_prob_test = gnb.predict_proba(X_test)
        predict_test = gnb.predict(X_test)
        
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, predict_test).ravel()
        
        accuracy = accuracy_score(y_test, predict_test)
        precision = precision_score(y_test, predict_test)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        
        tnr = specificity
        fpr = 1 - specificity
        fnr = 1 - tpr
        tpr = sensitivity
        
        
        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        tnrs.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)
        tprs.append(tpr)
    mean_tnr, std_tnr, min_tnr, max_tnr = np.mean(tnrs), np.std(tnrs), np.min(tnrs), np.max(tnrs)
    mean_fpr, std_fpr, min_fpr, max_fpr = np.mean(fprs), np.std(fprs), np.min(fprs), np.max(fprs)
    mean_fnr, std_fnr, min_fnr, max_fnr = np.mean(fnrs), np.std(fnrs), np.min(fnrs), np.max(fnrs)
    mean_tpr, std_tpr, min_tpr, max_tpr = np.mean(tprs), np.std(tprs), np.min(tprs), np.max(tprs)
    mean_accuracy, std_accuracy , min_accuracy, max_accuracy = np.mean(accuracies), np.std(accuracies), np.min(accuracies), np.max(accuracies)
    mean_precision,std_precision, min_precision, max_precision = np.mean(precisions), np.std(precisions), np.min(precisions), np.max(precisions)
    mean_sensitivity, std_sensitivity, min_sensitivity, max_sensitivity = np.mean(sensitivities), np.std(sensitivities), np.min(sensitivities), np.max(sensitivities)
    mean_specificity, std_specificity, min_specificity, max_specificity = np.mean(specificities), np.std(specificities), np.min(specificities), np.max(specificities)
    
    rates = np.array([[mean_tnr, std_tnr, min_tnr, max_tnr], [mean_fpr, std_fpr, min_fpr, max_fpr],
                     [mean_fnr, std_fnr, min_fnr, max_fnr], [mean_tpr, std_tpr, min_tpr, max_tpr]])
    apss = np.array([[mean_accuracy, std_accuracy, min_accuracy, max_accuracy],[mean_precision,std_precision, min_precision, max_precision],
                     [mean_sensitivity, std_sensitivity, min_sensitivity, max_sensitivity], [mean_specificity, std_specificity, min_specificity, max_specificity]])
    if mean_sensitivity > 0.65 : break
#%% Confusion Matrix with majority
while True:
    tprs = [] #true positive rates
    aucs = [] #areas under curves
    
    k = 6
    chs = [29,7,24]
    knn = KNeighborsClassifier(k)
    gnb = GaussianNB()
    tnrs, fprs, fnrs, tprs = [],[],[],[]
    accuracies = []
    precisions = []
    specificities = []
    sensitivities = []
    f = [0,1,4,5,8]
    
    for p in generator62(): #On fait 10 essais puis on moyenne les résultats des essais
        #Séparation des patients en train et test:
        predict_tests = []
        for ch in chs:
            X_train_patient, X_test_patient = empty_array.copy(), empty_array.copy()
            y_train_patient, y_test_patient = [],[]
            #X_train_patient, y_train_patient, X_valid_patient, y_valid_patient, X_test_patient, y_test_patient = [],[],[],[],[],[]
            train_list = [0,1,2,3,4,5,6,7]
            train_list.remove(p[0]) ; train_list.remove(p[1]) ;
            
            for i in range(6) : #On ajoute à la suite les features de chaque channel de chaque patient de la train list
                X_train_patient = np.vstack((X_train_patient, X_patient[train_list[i]][ch]))
                y_train_patient = np.concatenate((y_train_patient, y_patient[train_list[i]][ch]))
            for i in range(2) :
                X_test_patient = X_patient[p[i]][ch]
                y_test_patient = y_patient[p[i]][ch]
            
            X_train , y_train = X_train_patient[:,np.array(f)], y_train_patient
            X_test  , y_test  = X_test_patient[:,np.array(f)], y_test_patient
        
            
            #if X_train.shape[0] * X_test.shape[0] == 0 : continue ##Si l'un des trois est vide à cause de channels exclus
            
            #Entrainement pour une répartition aléatoire train/test sur les n :
            gnb.fit(X_train, y_train)
            predict_prob_test = gnb.predict_proba(X_test)
            predict_test = gnb.predict(X_test)
            predict_tests.append(predict_test)
        #On regarde les prédictions sur 3 canaux et on prend les 2 qui sont d'accords
        majority_predict = np.zeros(y_test.shape[0])
        for test in range(y_test.shape[0]):
            predictions = [predict_tests[0][test], predict_tests[1][test], predict_tests[2][test]]
            if predictions.count(1) >= 2 : majority_predict[test] = 1;
            else : majority_predict[test] = 0; #0 étant endormi
            
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, majority_predict).ravel()
        
        accuracy = accuracy_score(y_test, predict_test)
        precision = precision_score(y_test, predict_test)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        
        tnr = specificity
        fpr = 1 - specificity
        fnr = 1 - tpr
        tpr = sensitivity
        
        
        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        tnrs.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)
        tprs.append(tpr)
    mean_tnr, std_tnr, min_tnr, max_tnr = np.mean(tnrs), np.std(tnrs), np.min(tnrs), np.max(tnrs)
    mean_fpr, std_fpr, min_fpr, max_fpr = np.mean(fprs), np.std(fprs), np.min(fprs), np.max(fprs)
    mean_fnr, std_fnr, min_fnr, max_fnr = np.mean(fnrs), np.std(fnrs), np.min(fnrs), np.max(fnrs)
    mean_tpr, std_tpr, min_tpr, max_tpr = np.mean(tprs), np.std(tprs), np.min(tprs), np.max(tprs)
    mean_accuracy, std_accuracy , min_accuracy, max_accuracy = np.mean(accuracies), np.std(accuracies), np.min(accuracies), np.max(accuracies)
    mean_precision,std_precision, min_precision, max_precision = np.mean(precisions), np.std(precisions), np.min(precisions), np.max(precisions)
    mean_sensitivity, std_sensitivity, min_sensitivity, max_sensitivity = np.mean(sensitivities), np.std(sensitivities), np.min(sensitivities), np.max(sensitivities)
    mean_specificity, std_specificity, min_specificity, max_specificity = np.mean(specificities), np.std(specificities), np.min(specificities), np.max(specificities)
    
    rates = np.array([[mean_tnr, std_tnr, min_tnr, max_tnr], [mean_fpr, std_fpr, min_fpr, max_fpr],
                     [mean_fnr, std_fnr, min_fnr, max_fnr], [mean_tpr, std_tpr, min_tpr, max_tpr]])
    apss = np.array([[mean_accuracy, std_accuracy, min_accuracy, max_accuracy],[mean_precision,std_precision, min_precision, max_precision],
                     [mean_sensitivity, std_sensitivity, min_sensitivity, max_sensitivity], [mean_specificity, std_specificity, min_specificity, max_specificity]])
    if mean_sensitivity > 0.65 : break
#%% 3D plots
X_backup = X_all_patients.copy()
y_backup = y_all_patients.copy()
for ch in range (29,30):
    X_all_patients_asleep[ch] = np.vstack((X_all_patients_asleep[ch], X_all_patients[ch][y_all_patients[ch] == 2]))
    X_all_patients_awake[ch] = np.vstack((X_all_patients_awake[ch], X_all_patients[ch][y_all_patients[ch] == 1]))
#        if y_all_patients[ch][i] == 2:
#            X_all_patients_asleep[ch] = np.concatenate(X_all_patients_asleep[ch], X_all_patients[ch][i])
#        elif y_all_patients[ch][i] == 1:
#            X_all_patients_awake[ch] = np.concatenate(X_all_patients_awake[ch], X_all_patients[ch][i])
feature_x = 0
feature_y = 1
feature_z = 7
for ch in range(29,30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n = 100
    
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    xs = X_all_patients_asleep[ch][::20,feature_x]
    ys = X_all_patients_asleep[ch][::20,feature_y]
    zs = X_all_patients_asleep[ch][::20,feature_z]
    ax.scatter(xs, ys, zs, c='b', marker='.', label = 'asleep')
    xs = X_all_patients_awake[ch][::20,feature_x]
    ys = X_all_patients_awake[ch][::20,feature_y]
    zs = X_all_patients_awake[ch][::20,feature_z]
    ax.scatter(xs, ys, zs, c='orange', marker = '.', label = 'awake')
    ax.legend()    
    ax.set_xlabel(features_names[feature_x])
    ax.set_ylabel(features_names[feature_y])
    ax.set_zlabel(features_names[feature_z])
    
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
