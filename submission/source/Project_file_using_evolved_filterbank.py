#!/usr/bin/env python
# coding: utf-8

# In[20]:


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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from pycm import *
from sklearn.svm import SVC
import time
from sklearn.utils import shuffle
import csaps
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features.sigproc import preemphasis
from python_speech_features.base import delta
from python_speech_features.base import fbank
from python_speech_features.sigproc import framesig
from termcolor import colored
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics


# In[13]:


# fixed points taken for spline interpolation
xv=[0,0.35,0.69,1]    

# FIXED PARAMETER FOR endSPLINE 2
r_1=np.random.rand(4)
x2_fixed=np.linspace(0,1,6)
x2_modified=np.asarray(x2_fixed)

total_fbank = 30
nf=26

# nf equidistant point in between 0 & 1
xx=np.linspace(0,1,nf+2)              # will replace xx_full

yv=np.zeros((total_fbank,4))                 # initialising y values for spline 1
#y2_rand=np.zeros((total_fbank,4))            # initialising y values for spline 2
y2_rand=np.zeros((total_fbank,6))

sigma=np.zeros(total_fbank)           # 1-D arrray
rho=np.zeros(total_fbank)

EA_population=np.zeros((total_fbank,10))           # random EA population


# In[14]:


for fb in range (total_fbank):                               # total_filterbank
       
    #PARAMETER INITIALIZATION FOR SPLINE 1   
    a=0.1                                            #parameter to limit range of spline in bet. 0 & 1   
    y1=a+random()*(1-2*a)
    delta=random()*(1-a-y1)
    y2=y1+delta
    
    yv[fb][:]=[0,y1,y2,1]
    
    # 1st derivative calculation at end points
    s1 = xv[0]-xv[1]         # sigma is the 1st derivative at x=0,y=0
    s2 = xv[0]-xv[2]
    s3 = xv[0]-xv[3]
    s12 = s1-s2
    s13 = s1-s3 
    s23 = s2-s3
    sigma[fb]=-(s1*s2/(s13*s23*s3))*yv[fb][3]+(s1*s3/(s12*s2*s23))*yv[fb][2]-(s2*s3/(s1*s12*s13))*yv[fb][1]+(1./s1+1./s2+1./s3)*yv[fb][0]
    
    s11 = xv[3]-xv[2]         #rho is the 1st derivative at x=1,y=1
    s22 = xv[3]-xv[1]
    s33 = xv[3]-xv[0]
    s_12 = s11-s22 
    s_13 = s11-s33 
    s_23 = s22-s33
    rho[fb]=-(s11*s22/(s_13*s_23*s33))*yv[fb][0]+(s11*s33/(s_12*s22*s_23))*yv[fb][1]-(s22*s33/(s11*s_12*s_13))*yv[fb][2]+(1./s11+1./s22+1./s33)*yv[fb][3]
    
    # PARAMETER INTIALIZATION FOR SPLINE 2
    y2_rand[fb,:]= np.random.rand(6)*(0.9-0.25)+0.25
    
    #cc=[y1, delta, sigma[0,fb], rho[0,fb]]+ y2_rand[fb,:].tolist()
    EA_population[fb,:]=[y1, delta, sigma[fb], rho[fb]]+ y2_rand[fb,:].tolist()


# In[15]:


fs=8000

X_all = []
Y_all = []

path = os.getcwd()
list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
path = list_subfolders_with_paths[1]

for filename in glob.glob(os.path.join(path, '*.wav')):
    data, sampling_rate = librosa.load(filename,sr=None)
    X_all.append(data)
    
    if ('W' in filename):
        Y_all.append(0)  
    elif('L' in filename):
        Y_all.append(1)
    elif('E' in filename):
        Y_all.append(2)
    elif('A' in filename):
        Y_all.append(3)
    elif('F' in filename):
        Y_all.append(4)
    elif('T' in filename):
        Y_all.append(5)
    else:                                #(filename[22]=='N')
        Y_all.append(6)
        
X_pre_emp = []
N_all = len(X_all)

for i in range(N_all):
    X_pre_emp.append(preemphasis(X_all[i], coeff=0.94))


# In[21]:


##  RUNNING FROM THE PREVIOUS RESULT

from python_speech_features.base import delta


t1 = time.perf_counter()

from collections import OrderedDict
old_settings = np.seterr(all='print')
OrderedDict(np.geterr())

gen = 1                                 # no. of generation
g = 0


test_accuracy = np.zeros((total_fbank,gen))
test_accuracy_shuff = np.zeros((total_fbank, gen))
test_accuracy_noshuff = np.zeros((total_fbank, gen))

while(g<gen):
    
    for fb in range (total_fbank):                                           # fb=1:total_fbank
        
        # formation of spline 1
        y_spline1 = splineinterpolation(xv,yv[fb][:],EA_population[fb][2], EA_population[fb][3])  

        #plot(xx_mod,y_spline1)  
        y_min= min(y_spline1)                      
        y_max= max(y_spline1)

        ## OPTIMAIZATION OF FITER FREQ. LOCATION 
        freq=np.zeros(nf+1)
        for i in range (nf+1):                                   # i = 1:nf+1                            
            freq[i] = (y_spline1[i]-y_min)*fs/(2*(y_max-y_min))

        freq=np.sort(freq)

        ## OPTIMIZATION OF FILTER AMPLITUDE
        
        y2_modified = y2_rand[fb,:].tolist()
        cs=csaps.CubicSmoothingSpline(x2_modified, y2_modified, smooth=1)
        y_spline2=cs(xx)               
        #plt.plot(x2_modified, y2_modified, 'o', label='data')
        #plt.plot(xx, cs(xx), label="S")

        maxi=0
        for i in range(nf-1):                                     #i=1:nf-1
            t=np.absolute(freq[i+2]-freq[i])
            if(t > maxi):
                maxi=t

        N=256
        col=math.ceil(maxi*N/fs)+10
        filters=np.zeros((26,col))

        BW=np.zeros(nf)
        for i in range (nf):                                         # i=1:nf
            BW[i]=6.23*pow(freq[i]/1000,2)+93.39*(freq[i]/1000)+28.52

        #plt.figure(figsize=(3,3))

        for i in range (nf):                                               # i=1:nf
            yline1=[]
            yline2=[]
            ff1=freq[i]-BW[i]/2
            ff2=freq[i]+BW[i]/2
            if(ff1>0):
                f1=np.arange(ff1,freq[i]+1,fs/N)                             #ff1:fs/N:freq(i)
            else:
                f1=np.arange(0,freq[i]+1,fs/N)                              #0:fs/N:freq[i]

            if(ff2>(fs/2)):
                f2=np.arange(freq[i],(fs/2)+1,fs/N)                         #freq(i):fs/N:(fs/2);
                freq_inter=np.intersect1d(f2,fs/2)
                if(len(freq_inter)==0):
                    f2=np.append(f2,fs/2)

            else:
                f2=np.arange(freq[i],ff2+1,fs/N)                                      #freq(i):fs/N:ff2;
                freq_inter=np.intersect1d(f2,ff2)
                if(len(freq_inter)==0):
                    f2=np.append(f2,ff2)

            if(freq[i]==0 or freq[i]==fs/2):
                if(freq[i]==0):
                    yline1.append(np.array([0]))
                    yline2.append(-y_spline2[i]*(f2-ff2)/(ff2-freq[i]))
                else:
                    yline1.append(-y_spline2[i]*(f1-ff1)/(ff1-freq[i]))
                    yline2.append(np.array([0]))

            elif(ff1<0 and freq[i]!=0 and ff2 <= fs/2 ):
                yline1.append(y_spline2[i]*f1/freq[i])
                yline2.append(-y_spline2[i]*(f2-ff2)/(ff2-freq[i]))

            elif(ff2 > (fs/2) and freq[i] < (fs/2) and ff1>0):
                yline1.append(-y_spline2[i]*(f1-ff1)/(ff1-freq[i]))
                yline2.append(-y_spline2[i]*(f2-(fs/2))/((fs/2)-freq[i]))
            else:
                yline1.append(-y_spline2[i]*(f1-ff1)/(ff1-freq[i]))
                yline2.append(-y_spline2[i]*(f2-ff2)/(ff2-freq[i]))

            L1=yline1[0].tolist()
            L2=yline2[0].tolist()
            yline= L1+L2

            k_len=len(yline)
            f_final=np.append(f1,f2)
            filters[i,0:k_len]=yline

            #plt.plot(f_final,filters[i,0:k_len])             # filters(i,1:k_len)

        #plt.plot(freq ,y_spline2[0:-1])                  # y_spline2(1:end-1)
        
        
        #y_noisy = AWGN_new(X_pre_emp, 10)
        y_noisy = X_pre_emp
        #X_set, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy = Mfcc_revised_1(y_noisy, fs,freq,filters)
        X_set = Mfcc_feature(y_noisy, fs,freq,filters)
             
        # METHOD-1
        # test_accuracy_shuff[fb,g], test_accuracy_noshuff[fb,g]  = parameter_tuning(X_set, Y_all)
        
        seed = 42
        X_shuffle, y_shuffle = shuffle(X_set, Y_all, random_state=seed)
        X_train,X_test,Y_train,Y_test = data_norm_and_split(X_shuffle,y_shuffle)
       
        # METHOD-3
        test_svc = SVC(C=100,cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)
        test_svc.fit(X_train, Y_train)
        test_accuracy[fb,g] = test_svc.score(X_test, Y_test)
        

    print(test_accuracy)
    
    if (g < gen-1):
        EA_population, yv, y2_rand = Evolution(EA_population,test_accuracy,yv,g,y2_rand)

    g=g+1

np.savetxt("fbank_accuracy.csv", np.around(test_accuracy*100,2), delimiter=',', encoding=None,header='test_accuracy')

    
## CHOOSING 4 BEST FILTERBANKS FROM THE POPULATION 

best_fbank = 4

ind_sort = np.argsort(test_accuracy[:,gen-1])
best4_ind = ind_sort[::-1][:4]                          ## select the last for ele. of ind_sort

ESFB=np.zeros((best_fbank,10))

for i in range(best_fbank):
    ESFB[i,:]=EA_population[best4_ind[i],:]

np.savetxt("EA_population.csv", ESFB, delimiter=',', encoding=None,header='EA_population')

accuracy_fbank_test = np.zeros(best_fbank)                          
accuracy_fbank=np.zeros((best_fbank,2))
yv_test=np.zeros((best_fbank,4))
confusion_mat = [] 

for bfb in range(best_fbank):                             #cc=[y1, delta, sigma[0,fb], rho[0,fb]]+ y2_rand[fb,:].tolist()
    yv_test[bfb,1]=ESFB[bfb,0]                            # yv[fb][:]=[0,y1,y2,1]
    yv_test[bfb,2]=ESFB[bfb,0]+ESFB[bfb,1]

yv_test[:,3]=1
yv2_test=ESFB[:,4:]


for fb in range(best_fbank):
    
    # formation of spline 1
    y_spline1_t = splineinterpolation(xv,yv_test[fb,:],ESFB[fb,2], ESFB[fb,3])  


    #plot(xx_mod,y_spline1)  

    y_min_t = min(y_spline1_t)                      
    y_max_t = max(y_spline1_t)

    ## OPTIMAIZATION OF FITER FREQ. LOCATION 

    freq_t=np.zeros(nf+1)
    for i in range (nf+1):                                   # i = 1:nf+1                            
        freq_t[i] = (y_spline1_t[i]-y_min_t)*fs/ (2*(y_max_t-y_min_t))

    freq_t=np.sort(freq_t)

    ## OPTIMIZATION OF FILTER AMPLITUDE

    y2_mod_t =  yv2_test[fb,:].tolist()
    cs_t = csaps.CubicSmoothingSpline(x2_modified, y2_mod_t,smooth=1)
    y_spline2_t=cs_t(xx)               # xx consists nf+2 pts. including end points

    #plt.plot(x2_modified, y2_modified, 'o', label='data')
    #plt.plot(xx, cs(xx), label="S")

    maxi_t=0
    for i in range(nf-1):                                     #i=1:nf-1
        t=np.absolute(freq_t[i+2]-freq_t[i])
        if(t > maxi_t):
            maxi_t=t

    N=256
    col_t=math.ceil(maxi_t*N/fs)+10
    filters_t=np.zeros((26,col_t))

    BW_t=np.zeros(nf)
    for i in range (nf):                                         # i=1:nf
        BW_t[i]=6.23*pow(freq_t[i]/1000,2)+93.39*(freq_t[i]/1000)+28.52


    plt.figure(figsize=(6,3))
    #plt.subplot(2,2)
    
    for i in range (nf):                                               # i=1:nf
        yline1_t = []
        yline2_t = []

        ff1_t = freq_t[i]-BW_t[i]/2
        ff2_t = freq_t[i]+BW_t[i]/2
        if(ff1_t > 0):
            f1_t = np.arange(ff1_t, freq_t[i]+1, fs/N)                             #ff1:fs/N:freq(i)
        else:
            f1_t = np.arange(0, freq_t[i]+1, fs/N)                                                        #0:fs/N:freq[i]

        if(ff2_t > (fs/2)):
            f2_t = np.arange(freq_t[i],(fs/2)+1, fs/N)                         #freq(i):fs/N:(fs/2);
            freq_inter_t = np.intersect1d(f2_t, fs/2)
            if(len(freq_inter_t) == 0):
                f2_t = np.append(f2_t, fs/2)

        else:
            f2_t = np.arange(freq_t[i], ff2_t+1, fs/N)                                      #freq(i):fs/N:ff2;
            freq_inter_t = np.intersect1d(f2_t, ff2_t)
            if(len(freq_inter_t) == 0):
                f2_t = np.append(f2_t, ff2_t)


        if(freq_t[i] == 0 or freq_t[i] == fs/2):
            if(freq_t[i]==0):
                yline1_t.append(np.array([0]))
                yline2_t.append(-y_spline2_t[i]*(f2_t-ff2_t)/(ff2_t-freq_t[i]))
            else:
                yline1_t.append(-y_spline2_t[i]*(f1_t-ff1_t)/(ff1_t-freq_t[i]))
                yline2_t.append(np.array([0]))

        elif(ff1_t<0 and freq_t[i]!=0 and ff2_t <= fs/2 ):
            yline1_t.append(y_spline2_t[i]*f1_t / freq_t[i])
            yline2_t.append(-y_spline2_t[i]*(f2_t-ff2_t) / (ff2_t-freq_t[i]))

        elif(ff2_t > (fs/2) and freq_t[i] < (fs/2) and ff1_t > 0):
            yline1_t.append(-y_spline2_t[i]*(f1_t-ff1_t) / (ff1_t-freq_t[i]))
            yline2_t.append(-y_spline2_t[i]*(f2_t-(fs/2)) / ((fs/2)-freq_t[i]))
        else:
            yline1_t.append(-y_spline2_t[i]*(f1_t-ff1_t) / (ff1_t-freq_t[i]))
            yline2_t.append(-y_spline2_t[i]*(f2_t-ff2_t) / (ff2_t-freq_t[i]))


        L1_t = yline1_t[0].tolist()
        L2_t = yline2_t[0].tolist()
        yline_t = L1_t+L2_t

        k_len_t = len(yline_t)
        f_final_t = np.append(f1_t,f2_t)
        filters_t[i,0:k_len_t] = yline_t
        plt.plot(f_final_t, filters_t[i,0:k_len_t], color='k')             
        
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Gain')
    plt.ylim(0,1)
    plt.xlim(0,4000)
    plt.savefig('best_fbank {0}.jpg'.format(fb) ,bbox_inches ='tight' )
   
    plt.plot(freq_t , y_spline2_t[0:-1], color = 'r')                
    plt.savefig('spline & best_fbank {0}.jpg'.format(fb), bbox_inches ='tight' )
    plt.close()

    
    print(colored("The observation is made for the best filter bank {}".format (fb), 'red', 'on_grey'), "\n")

    best_feature_set = Mfcc_feature(X_pre_emp, fs, freq_t, filters_t)
    seed = 42
    X_shuff, y_shuff = shuffle(best_feature_set, Y_all, random_state=seed)
    X_train_t,X_test_t,Y_train_t,Y_test_t = data_norm_and_split(X_shuff,y_shuff)
    
    # METHOD-3
    test_svc_bf = SVC(C=100,cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
    test_svc_bf.fit(X_train_t, Y_train_t)
    accuracy_fbank_test[fb] = test_svc_bf.score(X_test_t, Y_test_t)
    
    y_predd = test_svc_bf.predict(X_test_t)
    confusion_mat.append(confusion_matrix(Y_test_t, y_predd))
    crr = metrics.classification_report(Y_test_t, y_predd)
    print(crr)
    
    
    """
    accuracy_fbank_test[fb], cf, cr = best_estimator(best_feature_set, Y_all)
    confusion_mat.append(cf)
    print(cr)
    """
    
np.savetxt("best_fbank_acc.csv", np.around(accuracy_fbank_test*100,2), delimiter=',', encoding=None,header='best_fbank_acc')

t2 = time.perf_counter()

print(f'Finished in {(t2-t1)/60} minutes')


# In[17]:


def Mfcc_feature(X_data, fs, freq, filters):
    
    N_data=len(X_data)
    fs=8000   ## fs=8 kHz
    #feat_size = 42

    #feature_vect = np.zeros((N_data, feat_size))

    mfcc_coeff = np.zeros((N_data, 13))
    #delta_coeff = np.zeros((N_data, 13))
    #delta_delta_coeff = np.zeros((N_data, 13))
    #frame_energy = np.zeros((N_data,3))
    
    N = 256
    frame_len = 0.03*fs
    M = 13          #13 point dct of a voiced frame signal
    nf = 26
    
    ## using a 30-ms Hamming window with 7.5-ms step.
    for i in range (len(X_data)):
       
        frames = framesig(X_data[i], frame_len, frame_step= 0.75*frame_len, winfunc=np.hamming)
        frame_num = len(frames)

        y_fft=[]
        energy = []
        avg_energy = 0
        for jk in range(frame_num):
            temp = np.dot(frames[jk],frames[jk])
            energy.append(temp)
            avg_energy += temp

        avg_energy = avg_energy/frame_num
        threshold = avg_energy*0.3
        #voiced_energy = []

        for fr in range(len(frames)):
            if (energy[fr] > threshold):                     #energy[q]/frame_len < 0.2, not dividing by frame_len
                #voiced_energy.append(energy[fr])
                temp_fft = np.absolute(np.fft.fft(frames[fr,:], N))
                y_fft.append(temp_fft*temp_fft)          

        #voiced_energy = np.asarray(voiced_energy)
        y_fft = np.asarray(y_fft)          # shape (-,256)
        y_fft = np.transpose(y_fft)

        #frame_energy[i,:] = [np.mean(voiced_energy), min(voiced_energy), max(voiced_energy) ]

        voiced_frame_no = y_fft.shape[1]   
        DCT_mat = np.zeros((voiced_frame_no,M))  

        for p in range (voiced_frame_no):        # p=1:frame_no                    
            E_spect=np.zeros(nf-3)          

            for q in range(nf-3):                # q=1:nf-3
                rr=np.arange(math.floor(freq[q]*N/fs)+1,math.ceil(freq[q+2]*N/fs)+1,1)   # rows of voiced frames

                E_spect[q]= filters[q,0:len(rr)].dot(y_fft[rr,p])
                if(E_spect[q] == math.nan or E_spect[q] == math.inf):
                    E_spect[q] = 0

            for ee in range (len(E_spect)):
                if(E_spect[ee]!=0 ):
                    E_spect[ee] = math.log(np.absolute(E_spect[ee]))
                else:
                    E_spect[ee] = 0

            DCT_mat[p,:]=np.absolute(scipy.fftpack.dct(E_spect,n=M))


        #delta_feat = delta(DCT_mat, N = len(DCT_mat))
        #delta_delta_feat = delta(delta_feat, N = len(DCT_mat))

        mfcc_coeff[i,:] = np.mean(DCT_mat,axis=0)
        #delta_coeff[i,:] = np.mean(delta_feat, axis = 0)
        #delta_delta_coeff[i,:] = np.mean(delta_delta_feat, axis = 0)

        #feature_vect[i,:] = np.hstack((mfcc_coeff[i,:], delta_coeff[i,:], delta_delta_coeff[i,:],
        #                                     frame_energy[i,:]))
    
    #return feature_vect, mfcc_coeff, delta_coeff, delta_delta_coeff, frame_energy
    return mfcc_coeff


# In[8]:


def data_norm_and_split(X_set,Y_data):
    Xx = X_set
    yy = np.asarray(Y_data)

    X_t, X_v, y_t, y_v = train_test_split(Xx,yy,test_size=0.2,random_state=27)

    scaler = preprocessing.StandardScaler()
    Xt_trans = scaler.fit_transform(X_t)
    Xv_trans = scaler.fit_transform(X_v)
    
    return Xt_trans,Xv_trans,y_t,y_v


def splineinterpolation(xv,yv,yp1,ypn):

    y2=np.zeros(4)         #2nd derivative
    n=len(y2)
    u=np.zeros(n-1)
    
    if (yp1 > 0.99e99):
        y2[0]=0
        u[0]=0
    else:
        y2[0]=yp1
        u[0]=(3.0/(xv[1]-xv[0]))*((yv[1]-yv[0])/(xv[1]-xv[0]-yp1))
    

    for i in range(1,n-1):                          #i=2:n-1
        sig=(xv[i]-xv[i-1])/(xv[i+1]-xv[i-1])
        p=sig*y2[i-1]+2.0
        y2[i]=(sig-1.0)/p
        u[i]=(yv[i+1]-yv[i])/(xv[i+1]-xv[i]) - (yv[i]-yv[i-1])/(xv[i]-xv[i-1])
        u[i]=(6.0*u[i]/(xv[i+1]-xv[i-1])-sig*u[i-1])/p
    

    if(ypn > 0.99e99):
        qn=0
        un=0
    else:
        qn=0.5
        un=(3.0/(xv[n-1]-xv[n-2]))*(ypn-(yv[n-1]-yv[n-2])/(xv[n-1]-xv[n-2]))
    
    y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0)
                                     
    for k in np.arange(n-2,-1,-1):                                        # k=n-1:-1:1
        y2[k]=y2[k]*y2[k+1]+u[k]
    

    nf=26
    jl=1
    klo=jl
    khi=jl+1
    xx=xv
    h=xx[khi]-xx[klo]
                                     
    #if(h==0) % throw warning
                                     
    x = np.linspace(0,1,nf+3)          # nf+1 points bt 0 & 1, excluding 0 & 1
    x=np.delete(x,[0])
    x=np.delete(x,[27])
                            
    y=np.zeros(nf+1)
    
    for i in range (nf+1):                                 # i=1:nf+1
        a=(xx[khi]-x[i])/h
        b=(x[i]-xx[klo])/h
        y[i]=a*yv[klo]+b*yv[khi]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6.0
        if(y[i]<0):
            y[i]=0
        if(y[i]>1):
            y[i]=1
        
    #x=[0,x,1];
    #y=[0,y,1];
    # plot(x,y);

    return y


def Evolution(EA_population,accuracy,yv,g,y2_rand):
    import random
    total_fbank=30
    current_member = 1
    lamda=20                                                                #selecting lamda individuals from 30 chromosomes
    chromo_size=10
    mating_pool=np.zeros((total_fbank,chromo_size))
    val =max(accuracy[:,g])
    best_indx=accuracy[:,g].tolist().index(max(accuracy[:,g]))
    best_individual=EA_population[best_indx,:]                             # best individual out of 30 individuals
    mating_pool[0,:]=best_individual                                       #elitist strategy
    #qq=np.zeros(lamda-1)
    qq=[]
    while current_member <lamda:
        rand_ind=random.sample(range(total_fbank), 10)                     # picking 10 individuals randomly out of 30
        best=0
        counter=0
        for j in range(len(rand_ind)):
            if current_member==1 and accuracy[rand_ind[j],g] > best and rand_ind[j]!=best_indx:
                best= accuracy[rand_ind[j],g]
                counter=j
            elif current_member >1 and accuracy[rand_ind[j],g] > best and rand_ind[j]!=best_indx:
                inter=list(set(qq) & set(rand_ind))
                if not inter:
                    best=accuracy[rand_ind[j],g]
                    counter=j
                else:
                    continue
               
        if (counter>0):
            mating_pool[current_member,:]=EA_population[rand_ind[counter],:]
            qq.append(rand_ind[counter])
            current_member+=1
      
    ## ONE POINT CROSSOVER
      
    slt=np.random.permutation(20) 
    select=slt.tolist()
    select.remove(0) 
    kk=20
    total=list(range(0,30))
    r=list(set(total).difference(set(qq)))
    if best_indx in r:
        r.remove(best_indx)
    n=len(r)
    
    for q in range(0,18,2):
        prob_crr = np.random.random()
        if (prob_crr <= 0.9):
            point=random.randint(1,8)
            mating_pool[kk,:]=mating_pool[select[q],0:point].tolist() + mating_pool[select[q+1],point:].tolist()
                                 
        else:
            p=np.random.randint(n)
            prnt_idx=r[p]
            r.remove(prnt_idx)
            n=n-1
            mating_pool[kk,:]=EA_population[prnt_idx,:]
                                  
    ### MUTATION OPERATOR ###
        for zz in range(chromo_size):
            if np.random.random() <= 0.12 :                                     #mutation probability is 0.12
                mating_pool[kk,zz]=np.random.random()
              
             
        kk=kk+1
        if (kk==29):
            mating_pool[kk,:]=[mating_pool[19,i] + mating_pool[18,i] for i in range(len(mating_pool[19,:]))] 
            #mating_pool[kk,:]=mating_pool[19,:]mating_pool[18,:]      # making last mating population as the combi of 19th & 20th mating population
        
    EA_population=mating_pool
    
    ## PARAMETER UPDATION
    for ff in range(total_fbank): 
        ## spline1 parameter updation
        a1=0.1
        yv[ff,1]=EA_population[ff,0]
        if yv[ff,1]>=1:
            yv[ff,1]=a1+np.random.random()*(1-2*a1)                      # parameter to limit range of spline in bet. 0 & 1   
            EA_population[ff,0]=yv[ff,1]
                                  
        y2_temp=EA_population[ff,0]+EA_population[ff,1]
        if (y2_temp >1):
            delta_temp=np.random.random()*(1-a-yv[ff,1])
            y2_temp=delta_temp+yv[ff,1]
            EA_population[ff,1]=delta_temp
                                  
        yv[ff,2]=y2_temp
         
        ### spline2 parameter updation
        y2_rand[ff,:]=EA_population[ff,4:]
        for i in range(6):
            if (y2_rand[ff,i]>0.9 or y2_rand[ff,i]<0.25):
                y2_rand[ff,i]=np.random.random()*(0.9-0.25)+0.25
                EA_population[ff,i+4]=y2_rand[ff,i]
          
              
    
    return EA_population,yv,y2_rand


# In[9]:


def best_estimator(feature_vect, Y_all):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(feature_vect)
    y = np.asarray(Y_all)

    seed = 42
    X_shuffle, y_shuffle = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X_shuffle,y_shuffle,test_size=0.2,random_state=27)
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)
  
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 10)
    grid.fit(X_train,y_train)
    #print("best_estimator :- ", grid.best_estimator_, "\n")
    best_svc = grid.best_estimator_
    best_svc.fit(X_train, y_train)
    y_pred = best_svc.predict(X_test)
    cf = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test, y_pred)
    print(cf)
    print(cr)
    best_accuracy = best_svc.score(X_test, y_test)

    return best_accuracy, cf, cr


# In[10]:


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


# In[11]:


def parameter_tuning(feature_vect, Y_all):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(feature_vect)
    y = np.asarray(Y_all)

    seed = 42
    X_shuffle, y_shuffle = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X_shuffle,y_shuffle,test_size=0.2,random_state=27)
       
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 10)
    grid.fit(X_train,y_train)
    print("best_estimator :- ", grid.best_estimator_, "\n")

    grid_predictions = grid.predict(X_test)
    print("confusion_matrix :-" , confusion_matrix(y_test,grid_predictions))
    print("classification_report :-" , classification_report(y_test,grid_predictions))
    print("score:- " , grid.score(X_test,y_test),"\n")

    scores_best = {"True":[], "False": []}
    shuffs = [True, False]

    for shuff in shuffs:
        print(colored("The observation is made when shuffle is {}".format (shuff), 'red', 'on_grey'), "\n")
        best_svc = grid.best_estimator_

        # k-fold cross validation
        cv = KFold(n_splits=10, random_state=42, shuffle = shuff)
        for train_index, test_index in cv.split(X):
            
            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)
            X_train_1, X_test_1, y_train_1, y_test_1 = X[train_index], X[test_index], y[train_index], y[test_index]
            best_svc.fit(X_train_1, y_train_1)
            if(shuff == True):
                scores_best["True"].append(best_svc.score(X_test_1, y_test_1))
            else:
                scores_best["False"].append(best_svc.score(X_test_1, y_test_1))

            print("\n ")
        
    Score_shuff_true=scores_best["True"]
    Score_shuff_false = scores_best["False"]    
    return np.mean(Score_shuff_true), np.mean(Score_shuff_false)


# In[ ]:




