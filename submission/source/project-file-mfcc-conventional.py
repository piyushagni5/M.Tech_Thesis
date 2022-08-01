#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pyaudio
import wave
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import random

from scipy.io.wavfile import read
import scipy

from random import random
import math
from numpy.linalg import inv
from scipy.linalg import toeplitz

import wave, os, glob

from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as pchip

import librosa

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from pycm import *
from sklearn.svm import SVC

import time

from termcolor import colored

from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features.sigproc import preemphasis
from python_speech_features.base import delta
from python_speech_features.base import fbank
from python_speech_features.sigproc import framesig

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import numpy as np

# Import GridsearchCV from Scikit Learn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

from sklearn.utils import shuffle


# ### Pre-emphasis of the speech signal
# alpha = 0.94
# 
# X_pre_emp = X_data
# 
# for i in range(len(X_data)):
#     
#     for j in range(1,len(X_data[i])):
#         
#         X_pre_emp[i][j]=X_data[i][j]-0.94*X_data[i][j-1]
# 

# In[147]:


## SPLITTING TRAINING & TESTING DATASET 

fs=8000

X_data = []
Y_data = []

path = 'C:/emodb_new/wav'
for filename in glob.glob(os.path.join(path, '*.wav')):
    
    data, sampling_rate = librosa.load(filename,sr=None)
    X_data.append(data)
    
    if (filename[22]=='W'):
        Y_data.append(0)                 # W-->0--> Anger
    elif(filename[22]=='L'):
        Y_data.append(1)                 # L-->1--> Boredom
    elif(filename[22]=='E'):
        Y_data.append(2)                 # E-->2--> Disgust
    elif(filename[22]=='A'):
        Y_data.append(3)                 # A-->3--> Anxiety/fear
    elif(filename[22]=='F'):
        Y_data.append(4)                 # F-->4--> Happiness
    elif(filename[22]=='T'):
        Y_data.append(5)                 # T-->5-->sadness
    else:                                   
        Y_data.append(6)                 # N-->6--> Neutral
        


# #### python_speech_features.base.mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=<function <lambda>>)
# 1. signal – the audio signal from which to compute features. Should be an N*1 array
# 2. samplerate – the samplerate of the signal we are working with.
# 3. winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
# 4. winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
# 5. numcep – the number of cepstrum to return, default 13
# 6. nfilt – the number of filters in the filterbank, default 26.
# 7. nfft – the FFT size. Default is 512.
# 8. lowfreq – lowest band edge of mel filters. In Hz, default is 0.
# 9. highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
# 10. preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
# 11. ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
# 12. appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
# 13. winfunc – the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming

# #### inbuilt pre-emphasis function

# In[148]:


X_pre_emphasis = []

for i in range(len(X_data)):
    X_pre_emphasis.append(preemphasis(X_data[i], coeff=0.94))


# In[149]:


def feature_vector(X_data):    
    N_data=len(X_data)
    fs=8000   
    feat_size = 42

    feature_vect = np.zeros((N_data, feat_size))

    mfcc_coeff = np.zeros((N_data, 13))
    delta_coeff = np.zeros((N_data, 13))
    delta_delta_coeff = np.zeros((N_data, 13))
    frame_energy = np.zeros((N_data,3))

    ## using a 30-ms Hamming window with 7.5-ms step.
    for i in range (len(X_data)):

        mfcc_feat = mfcc( X_data[i], samplerate = 8000, winlen=0.03, winstep=0.0075, winfunc=np.hamming) 
        delta_feat = delta(mfcc_feat, N = mfcc_feat.shape[0])
        delta_delta_feat = delta(delta_feat, N = mfcc_feat.shape[0])
        #fbank_feat = logfbank(sig,rate)

        mfcc_coeff[i,:] = np.mean(mfcc_feat, axis = 0 )
        delta_coeff[i,:] = np.mean(delta_feat, axis = 0)
        delta_delta_coeff[i,:] = np.mean(delta_delta_feat, axis = 0)

        ## calculating frame energies, frames_size = no. of frames * frame_len

        frame_len = 0.03*fs
        frames = framesig(X_data[i], frame_len, frame_step= 0.75*frame_len, winfunc=np.hamming)
        energy = np.zeros(len(frames))
        
        for fr in range(len(frames)):
            energy[fr] = np.dot(frames[fr],frames[fr])
            

        max_frame = max(energy)
        min_frame = min(energy)
        avg_frame = np.mean(energy)

        frame_energy[i,:] = [avg_frame, min_frame, max_frame]

        feature_vect[i,:] = np.hstack((mfcc_coeff[i,:], delta_coeff[i,:], delta_delta_coeff[i,:],
                                     frame_energy[i,:]))
        
    return feature_vect, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy


# ### <font color='cyan'>Part-1 Evaluation of noisy speech signal in all SNR values</font> 

# In[150]:


def AWGN_new(X,snr):
    N = len(X)
    y_noisy = []
    for i in range(N): 
        x=X[i]
        sig_power = np.mean(np.abs(x.tolist())**2)
        sig_db = 10 * np.log10(sig_power)
        noise_db = sig_db -snr
        noise_power = 10 ** (noise_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_power), len(x))
        # Noise up the original signal
        y_noisy.append(x + noise)
    return y_noisy


# In[21]:


tt1 = time.perf_counter()

SNR_dB = [0,10,20,30,'clean']
SNR_dB_scores = np.zeros((len(SNR_dB), 2))

count = 0

for snr in SNR_dB:
    print(colored("The observation is made for {} SNR_dB".format (snr), 'red', 'on_grey'), "\n")

    if snr!= 'clean':
        X_data_noisy = AWGN_new(X_data, snr)
    else:
        X_data_noisy = X_data
    
    
    feature_vect_noisy, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy = feature_vector(X_data_noisy)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_noisy = scaler.fit_transform(feature_vect_noisy)
    y_noisy = np.asarray(Y_data)
    
    seed = 42
    X_shuffle, y_shuffle = shuffle(X_noisy, y_noisy, random_state=seed)
    
    ## test train split
    X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(X_shuffle,y_shuffle,test_size=0.2,random_state=27)
    
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    
    grid_1 = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 5)
    grid_1.fit(X_train_noisy,y_train_noisy)

    print(grid_1.best_estimator_)       ## found the best estimator using grid search

    grid_1_predictions = grid_1.predict(X_test_noisy)
    print(confusion_matrix(y_test_noisy,grid_1_predictions))
    print(classification_report(y_test_noisy,grid_1_predictions))
    print(grid_1.score(X_test_noisy,y_test_noisy))

    shuffs_noisy = [True, False]
    scores_best_noisy = {"True":[], "False": []}

    for shuff_noisy in shuffs_noisy:
        print(colored("The observation is made when shuffle is {}".format (shuff_noisy), 'red', 'on_grey'), "\n")
        best_svc_noisy = grid_1.best_estimator_

        # k-fold cross validation
        cv_1 = KFold(n_splits=10, random_state=42, shuffle = shuff_noisy)

        for train_index, test_index in cv_1.split(X_noisy):
            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)

            X_train_1, X_test_1, y_train_1, y_test_1 = X_noisy[train_index], X_noisy[test_index], y_noisy[train_index], y_noisy[test_index]
            best_svc_noisy.fit(X_train_1, y_train_1)

            if(shuff_noisy == True):
                scores_best_noisy["True"].append(best_svc_noisy.score(X_test_1, y_test_1))
            else:
                scores_best_noisy["False"].append(best_svc_noisy.score(X_test_1, y_test_1))

            print("\n ")

    SNR_dB_scores[count,0] =np.mean(scores_best_noisy["True"])
    SNR_dB_scores[count,1] = np.mean(scores_best_noisy["False"]) 
    count += 1

tt2 = time.perf_counter()
print(f'Finished in {(tt2-tt1)/60} minutes')


# In[22]:


SNR_dB_scores


# In[13]:


df_noisy = pd.DataFrame({'SNR_dB':[0,10,20,30,'clean'], 'CV_acc_shuff_T': SNR_dB_scores[:,0],
                   'CV_acc_shuff_F':SNR_dB_scores[:,1] })
df_noisy.head()


# In[ ]:


pd.concat([rs, ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])


# ### <font color='cyan'>Part-2 Evaluation of noisy pre_emphasis speech signal in all SNR values</font> 

# In[79]:


t1 = time.perf_counter()

SNR_dB_pre = [0,10,20,30,'clean']
SNR_dB_scores_pre = np.zeros((len(SNR_dB), 2))

count_pre = 0

for snr_pre in SNR_dB_pre:
    print(colored("The observation is made for {} SNR_dB".format (snr_pre), 'red', 'on_grey'), "\n")

    if snr_pre!= 'clean':
        X_data_noisy = AWGN_new(X_pre_emphasis, snr_pre)
    else:
        X_data_noisy = X_pre_emphasis
    feature_pre_emp, mfcc_coeff_pre_emp, delta_coeff_pre_emp, delta_delta_coeff_pre_emp, frame_energy_pre_emp = feature_vector(X_data_noisy)
    
    X_pre_emp = scaler.fit_transform(feature_pre_emp)
    y_pre_emp = np.asarray(Y_data)

    X_train_pre_emp, X_test_pre_emp, y_train_pre_emp, y_test_pre_emp = train_test_split(X_pre_emp,y_pre_emp,test_size=0.2,random_state=27)
    
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    grid_pre_emp = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 10)
    grid_pre_emp.fit(X_train_pre_emp,y_train_pre_emp)

    print(colored("The Hyper-parameter tuning for pre-emphasis speech signal", 'green', 'on_yellow'))
    print(grid_pre_emp.best_estimator_)

    grid_predictions_pre_emp = grid_pre_emp.predict(X_test_pre_emp)
    print(confusion_matrix(y_test_pre_emp, grid_predictions_pre_emp))
    print(classification_report(y_test_pre_emp, grid_predictions_pre_emp))
    print(grid_pre_emp.score(X_test_pre_emp,y_test_pre_emp))

    scores_best_pre_emp = {"True":[], "False": []}
    shuffs_pre_emp = [True, False]

    for shuff_pre_emp in shuffs_pre_emp:
        print(colored("The observation is made pre-emphasis when shuffle is {}".format (shuff), 'red', 'on_grey'), "\n")

        best_svc_pre_emp = grid_pre_emp.best_estimator_

        # k-fold cross validation

        cv_pre_emp = KFold(n_splits=10, random_state=42, shuffle = shuff_pre_emp)
        for train_index, test_index in cv_pre_emp.split(X):

            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)

            X_train_pre_emp, X_test_pre_emp, y_train_pre_emp, y_test_pre_emp = X_pre_emp[train_index], X_pre_emp[test_index], y_pre_emp[train_index], y_pre_emp[test_index]

            best_svc_pre_emp.fit(X_train_pre_emp, y_train_pre_emp)
            if(shuff_pre_emp == True):
                scores_best_pre_emp["True"].append(best_svc_pre_emp.score(X_test_pre_emp, y_test_pre_emp))
            else:
                scores_best_pre_emp["False"].append(best_svc_pre_emp.score(X_test_pre_emp, y_test_pre_emp))

            print("\n ")

    SNR_dB_scores_pre[count_pre,0] = np.mean(scores_best_pre_emp["True"])
    SNR_dB_scores_pre[count_pre,1] = np.mean(scores_best_pre_emp["False"]) 
    count_pre += 1

t2 = time.perf_counter()
print(f'Finished in {(t2-t1)/60} minutes')


# In[80]:


SNR_dB_scores_pre


# In[81]:


df_noisy_pre = pd.DataFrame({'SNR_dB':[0,10,20,30,'clean'], 'CV_acc_shuff_T': SNR_dB_scores_pre[:,0],
                   'CV_acc_shuff_F':SNR_dB_scores_pre[:,1] })
df_noisy_pre.head()


# ### <font color='cyan'>Part-3 Evaluation of noisy speech signal in all SNR values using only MFCC features</font> 

# In[26]:


tt1 = time.perf_counter()

SNR_dB = [0,10,20,30,'clean']
SNR_dB_scores_mfcc = np.zeros((len(SNR_dB), 2))

count_mfcc = 0

for snr_mfcc in SNR_dB:
    print(colored("The observation is made for {} SNR_dB".format (snr_mfcc), 'red', 'on_grey'), "\n")

    if snr_mfcc!= 'clean':
        X_data_noisy = AWGN_new(X_data, snr_mfcc)
    else:
        X_data_noisy = X_data
    
    feature_vect_noisy, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy = feature_vector(X_data_noisy)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_noisy = scaler.fit_transform(mfcc_coeff)
    y_noisy = np.asarray(Y_data)
    
    seed = 42
    X_shuff_noisy, y_shuff_noisy = shuffle(X_noisy, y_noisy, random_state=seed)
    ## test train split
    X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(X_shuff_noisy, y_shuff_noisy,test_size=0.2,random_state=27)
    grid_1 = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 5)
    grid_1.fit(X_train_noisy,y_train_noisy)

    print(grid_1.best_estimator_)       ## found the best estimator using grid search

    grid_1_predictions = grid_1.predict(X_test_noisy)
    print(confusion_matrix(y_test_noisy,grid_1_predictions))
    print(classification_report(y_test_noisy,grid_1_predictions))
    print(grid_1.score(X_test_noisy,y_test_noisy))

    shuffs_noisy = [True, False]
    scores_best_noisy = {"True":[], "False": []}

    for shuff_noisy in shuffs_noisy:
        print(colored("The observation is made when shuffle is {}".format (shuff_noisy), 'red', 'on_grey'), "\n")
        best_svc_noisy = grid_1.best_estimator_

        # k-fold cross validation
        cv_1 = KFold(n_splits=10, random_state=42, shuffle = shuff_noisy)

        for train_index, test_index in cv_1.split(X_noisy):
            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)

            X_train_1, X_test_1, y_train_1, y_test_1 = X_noisy[train_index], X_noisy[test_index], y_noisy[train_index], y_noisy[test_index]
            best_svc_noisy.fit(X_train_1, y_train_1)

            if(shuff_noisy == True):
                scores_best_noisy["True"].append(best_svc_noisy.score(X_test_1, y_test_1))
            else:
                scores_best_noisy["False"].append(best_svc_noisy.score(X_test_1, y_test_1))

            print("\n ")

    SNR_dB_scores_mfcc[count_mfcc,0] =np.mean(scores_best_noisy["True"])
    SNR_dB_scores_mfcc[count_mfcc,1] = np.mean(scores_best_noisy["False"]) 
    count_mfcc += 1

tt2 = time.perf_counter()
print(f'Finished in {(tt2-tt1)/60} minutes')


# In[ ]:





# In[27]:


SNR_dB_scores_mfcc


# In[16]:


df_noisy_mfcc = pd.DataFrame({'SNR_dB':[0,10,20,30,'clean'], 'Mfcc CV_acc shuff_T': SNR_dB_scores_mfcc[:,0],
                   'Mfcc CV_acc shuff_F':SNR_dB_scores_mfcc[:,1] })
df_noisy_mfcc.head()


# ### <font color='cyan'>Part-4 Evaluation of noisy pre-emphasis speech signal in all SNR values using only MFCC features</font> 

# In[164]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

t1 = time.perf_counter()
target_names = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

SNR_dB_pre = [0,10,20,'clean']
SNR_dB_scores_pre_mfcc = np.zeros((len(SNR_dB_pre), 2))

count_pre_mfcc = 0
conf_matrix = []

for snr_pre_mfcc in SNR_dB_pre:
    print(colored("The observation is made for {} SNR_dB".format (snr_pre_mfcc), 'red', 'on_grey'), "\n")

    if snr_pre_mfcc!= 'clean':
        X_data_noisy = AWGN_new(X_pre_emphasis, snr_pre_mfcc)
    else:
        X_data_noisy = X_pre_emphasis
    feature_pre_emp, mfcc_coeff_pre_emp, delta_coeff_pre_emp, delta_delta_coeff_pre_emp, frame_energy_pre_emp = feature_vector(X_data_noisy)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_pre_emp = scaler.fit_transform(mfcc_coeff_pre_emp)
    y_pre_emp = np.asarray(Y_data)

    #seed = 42
    #X_shuffle_pre, y_shuffle_pre = shuffle(X_pre_emp, y_pre_emp, random_state=seed)
    #X_train_pre_emp, X_test_pre_emp, y_train_pre_emp, y_test_pre_emp = train_test_split(X_shuffle_pre,y_shuffle_pre,test_size=0.2,random_state=27)
    X_train_pre_emp, X_test_pre_emp, y_train_pre_emp, y_test_pre_emp = train_test_split(X_pre_emp, y_pre_emp, test_size=0.2,random_state=27)
    
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid_pre_emp = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 10)
    grid_pre_emp.fit(X_train_pre_emp,y_train_pre_emp);

    print(colored("The Hyper-parameter tuning for pre-emphasis speech signal", 'green', 'on_yellow'))
    print(grid_pre_emp.best_estimator_)

    #grid_predictions_pre_emp = grid_pre_emp.predict(X_test_pre_emp)
    #print(confusion_matrix(y_test_pre_emp, grid_predictions_pre_emp))
    #print(classification_report(y_test_pre_emp, grid_predictions_pre_emp))
    #print(grid_pre_emp.score(X_test_pre_emp,y_test_pre_emp))
    """
    y_pred = grid_pre_emp.best_estimator_.predict(X_test_pre_emp)
    
    cm = confusion_matrix(y_test_pre_emp, y_pred)
    # Normalise
    cmn = cm.astype('float')  
    cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5,5))
    ft = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    figure = ft.get_figure() 
    figure.savefig('confusion_mat {0} dB.png'.format(snr_pre_mfcc), dpi=400)
    """
    scores_best_pre_emp = {"True":[], "False": []}
    shuffs_pre_emp = [True, False]

    for shuff_pre_emp in shuffs_pre_emp:
        print(colored("The observation is made pre-emphasis when shuffle is {}".format (shuff_pre_emp), 'red', 'on_grey'), "\n")

        best_svc_pre_emp = grid_pre_emp.best_estimator_

        # k-fold cross validation

        cv_pre_emp = KFold(n_splits=10, random_state=42, shuffle = shuff_pre_emp)
        for train_index, test_index in cv_pre_emp.split(X_pre_emp):

            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)

            X_train_pre, X_test_pre, y_train_pre, y_test_pre = X_pre_emp[train_index], X_pre_emp[test_index], y_pre_emp[train_index], y_pre_emp[test_index]

            best_svc_pre_emp.fit(X_train_pre, y_train_pre)
            
            if(shuff_pre_emp == True):
                scores_best_pre_emp["True"].append(best_svc_pre_emp.score(X_test_pre, y_test_pre))
                y_predd = best_svc_pre_emp.predict(X_test_pre)
                conf_matrix.append(confusion_matrix(y_test_pre, y_predd))
            else:
                scores_best_pre_emp["False"].append(best_svc_pre_emp.score(X_test_pre, y_test_pre))

            print("\n ")

    SNR_dB_scores_pre_mfcc[count_pre_mfcc,0] = np.mean(scores_best_pre_emp["True"])
    SNR_dB_scores_pre_mfcc[count_pre_mfcc,1] = np.mean(scores_best_pre_emp["False"]) 
    count_pre_mfcc += 1

t2 = time.perf_counter()
print(f'Finished in {(t2-t1)/60} minutes')


# In[165]:


SNR_dB_scores_pre_mfcc


# In[166]:


len(conf_matrix)


# In[177]:


conf_make = np.array([[21,0,1,2,2,0,0],[1,8,0,1,0,3,1],[0,1,3,1,1,0,0],[1,0,0,14,1,1,1],[5,2,1,2,5,0,2],[0,1,0,1,0,13,1],[0,1,0,1,0,2,6]])

conf_make


# In[179]:


km = conf_matrix[0:10]
cf = np.zeros((7,7))
for i in range(7):
    for t in range(10):
        cf[i]+=km[t][i]
sum=0
dia = 0
for i in range (7):
    for j in range(7):
        sum+=cf[i][j]
        if i==j:
            dia += cf[i][j]


# In[180]:


cf = conf_make
sum=0
dia = 0
for i in range (7):
    for j in range(7):
        sum+=cf[i][j]
        if i==j:
            dia += cf[i][j]


# In[181]:


dia/sum


# In[183]:


## confusion mat for 0db shuffle true case
cf = (cf / cf.astype(np.float).sum(axis=1)) *100
cf


# In[184]:


target_names = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

#cmn = cm.astype('float')  
#cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(8,8))
ft = sns.heatmap(cf, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.show(block=False)
figure = ft.get_figure() 
figure.savefig('fb3_EBDCC_confusion_mat {0} dB.png'.format('0'), dpi=400)


# In[ ]:


### best estimator

from sklearn.metrics import confusion_matrix
import seaborn as sns
import sklearn.metrics as metrics

t1 = time.perf_counter()
target_names = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

SNR_dB_pre = [0,10,20,'clean']
best_est_pre_mfcc = np.zeros(len(SNR_dB_pre))

cnt = 0
conf_matrix = []


for snr_pre_mfcc in SNR_dB_pre:
    print(colored("The observation is made for {} SNR_dB".format (snr_pre_mfcc), 'red', 'on_grey'), "\n")

    if snr_pre_mfcc!= 'clean':
        X_data_noisy = AWGN_new(X_pre_emphasis, snr_pre_mfcc)
    else:
        X_data_noisy = X_pre_emphasis
    feature_pre_emp, mfcc_coeff_pre_emp, delta_coeff_pre_emp, delta_delta_coeff_pre_emp, frame_energy_pre_emp = feature_vector(X_data_noisy)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_pre_emp = scaler.fit_transform(mfcc_coeff_pre_emp)
    y_pre_emp = np.asarray(Y_data)


    #seed = 42
    #X_shuffle, y_shuffle = shuffle(X, y, random_state=seed)
    #X_train, X_test, y_train, y_test = train_test_split(X_shuffle,y_shuffle,test_size=0.2,random_state=27)
    X_train, X_test, y_train, y_test = train_test_split(X_pre_emp, y_pre_emp, test_size=0.2,random_state=27)
  
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 10)
    grid.fit(X_train,y_train)
    #print("best_estimator :- ", grid.best_estimator_, "\n")
    best_svc = grid.best_estimator_
    best_svc.fit(X_train, y_train)
    y_pred = best_svc.predict(X_test)
    cf = confusion_matrix(y_test,y_pred)
    cr = metrics.classification_report(y_test, y_pred)
    #print(cf)
    print(cr)
    best_est_pre_mfcc[cnt] = best_svc.score(X_test, y_test)
    conf_matrix.append(cf)
    cnt+=1
    

t2 = time.perf_counter()
print(f'Finished in {(t2-t1)/60} minutes')


# In[159]:


best_est_pre_mfcc


# In[98]:


pd.concat([df_noisy,df_noisy_pre,df_noisy_mfcc, df_noisy_mfcc_pre], axis=1, keys=['Noisy Speech using 42 dim. feature'],['Noisy pre_emphasis Speech using 42 dim. feature'],['Noisy Speech using 13 dim. mfcc feature'],['Noisy pre_emphasis Speech using 13 dim. mfcc feature'])


# In[100]:


pd.concat([df_noisy,df_noisy_pre,df_noisy_mfcc, df_noisy_mfcc_pre], axis=1,keys=['1','2','3','4'])


# ### <font color='cyan'> DATA ANALYSIS : PLOTTING SECTION </font>

# In[12]:


scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(feature_vect)
y = np.asarray(Y_data)

X_pre_emp = scaler.fit_transform(feature_pre_emp)
y_pre_emp = np.asarray(Y_data)


# In[ ]:


feature_vect_noisy, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy = feature_vector(X)

feature_pre_emp, mfcc_coeff_pre_emp, delta_coeff_pre_emp, delta_delta_coeff_pre_emp, frame_energy_pre_emp = feature_vector(X_pre_emp)


# In[11]:


X.shape


# In[44]:


mfcc_coeff.shape


# mfcc_coeff = scaler.fit_transform(mfcc_coeff)
# delta_coeff = scaler.fit_transform(delta_coeff)
# delta_delta_coeff = scaler.fit_transform(delta_delta_coeff)
# frame_energy = scaler.fit_transform(frame_energy)

# In[127]:


df = pd.DataFrame({'Column1': mfcc_coeff[:, 0], 'Column2': mfcc_coeff[:, 1], 'Column3': mfcc_coeff[:, 2], 'Column4': mfcc_coeff[:, 3],
                   'Column5': mfcc_coeff[:, 4], 'Column6': mfcc_coeff[:, 5], 'Column7': mfcc_coeff[:, 6],'Column8': mfcc_coeff[:, 7],
                   'Column9': mfcc_coeff[:, 8], 'Column10': mfcc_coeff[:, 9], 'Column11': mfcc_coeff[:, 10],'Column12': mfcc_coeff[:, 11],
                   'Column13': mfcc_coeff[:, 12],
                   'avg_energy': frame_energy[:,0], 'min_energy': frame_energy[:,1], 'max_energy': frame_energy[:,2],'emotion': y})


# In[128]:


df.head()


# In[129]:


df.iloc[:,[0,1,13,14,15,16]].head()              # row 1 is exclusive


# In[77]:


df.emotion.head()


# In[132]:


sns.pairplot(df.iloc[:,[13,14,15,16]], hue = 'emotion', palette = 'Dark2')


# In[200]:


k=df.iloc[:,[0,1,16]]
k=k[(k.emotion == 0) | (k.emotion == 4)]

k.head()


# In[201]:



X_analyse = k.iloc[:,[0,1]].to_numpy()
y_analyse = k.emotion.to_numpy()


# In[202]:


X_analyse.shape


# In[203]:


y_analyse.shape


# In[204]:


Xx_train, Xx_test, yy_train, yy_test = train_test_split(X_analyse,y_analyse,test_size=0.2,random_state=27)

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid']}

grid_analyse = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 5)
grid_analyse.fit(Xx_train,yy_train)


# In[205]:


print(grid_analyse.best_estimator_)


# ### Employing the best parameter estimator

# #### W-->0--> Anger
# #### L-->1--> Boredom
# #### E-->2--> Disgust
# #### A-->3--> Anxiety/fear
# #### F-->4--> Happiness
# #### T-->5-->sadness
# #### N-->6--> Neutral

# In[206]:


from mlxtend.plotting import plot_decision_regions


# Training a classifier
svm_analyse = grid_analyse.best_estimator_     # C (Regularisation)
svm_analyse.fit(X_analyse, y_analyse)

# Plotting decision regions
#plt.figure(figsize=(10,10))
plot_decision_regions(X_analyse, y_analyse, clf=svm_analyse, legend=2)

# Adding axes annotations


plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM on Iris_linear')
plt.show()


# In[189]:


EBDCC_F3 = [67.29, 71.96,75.49,74.77]
EBDCC_F3.reverse()

EBDCC_WF3 = [65.42,71.03,73.83,74.77]
EBDCC_WF3.reverse()

MFB = [63.94,69.15,72.88,74.57]
MFB.reverse()

snr = [0,10,20,30]
#snr = {'clean' : 0 , '20 dB': 10, '10 dB': 20, '0 dB': 30 }

fig, axs = plt.subplots(1, 1, figsize=(4, 5), sharey=True)
#plt.figure(figsize= (5,6))
axs[0].plot(snr,EBDCC_F3,)
axs[0].plot(snr,EBDCC_WF3)
axs[0].plot(snr,MFB)


fig.suptitle('Categorical Plotting')


# In[216]:


EBDCC_F3 = [67.29, 71.96,75.49,74.77]
EBDCC_F3.reverse()
EBDCC_WF3 = [65.42,71.03,73.83,74.77]
EBDCC_WF3.reverse()

#EBDCC_WF1 = [64.49 ,65.42 ,70.09 ,76.64]
#EBDCC_WF1.reverse()

MFB = [63.94,69.15,72.88,74.57]
MFB.reverse()
snr = ['clean', '20 dB', '10 dB', '0 dB']
#activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

fig, ax = plt.subplots(figsize=(6, 3))

ax.plot(snr, MFB, label="MFB",   c='k', linewidth=2, markersize=12,linestyle='dashed')

ax.plot(snr, EBDCC_WF3, label="EBDCC-WF3" ,linewidth=2, markersize=12,linestyle='-.')

ax.plot(snr, EBDCC_F3, label="EBDCC-F3", c='r', linewidth=2, markersize=12,linestyle=':')
#ax.plot(snr, EBDCC_WF1, label="EBDCC_WF1")

plt.xlabel('SNR (dB)')
plt.ylabel('10-Fold CV accuracy (%)')
ax.legend()
plt.ylim((60,80))
plt.xlim((0,3))
plt.savefig('comparing_results' ,bbox_inches ='tight' )

plt.show()


# In[187]:


import matplotlib.pyplot as plt

data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')


# In[ ]:




