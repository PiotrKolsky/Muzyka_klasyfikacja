#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import warnings
warnings.filterwarnings("ignore")
import os
import scipy.io.wavfile
import numpy as np
import pandas as pd
import joblib
import imp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


def fft_extracting():

    #genres = [e for e in os.listdir('./genres/') if os.path.isdir('./genres/'+e)]
    genres = ['classical', 'blues', 'jazz', 'pop', 'metal', 'disco']
    
    X = pd.DataFrame(); y = pd.DataFrame()
    for gen in genres:
        files_list = list(os.listdir('./genres/'+gen))
        for file in files_list[:]:
            sample_rate, music_freq = scipy.io.wavfile.read('./genres/'+gen+'/'+file)
            print('fft extracting {}'.format(file))
            fft_features = abs(np.fft.rfft(music_freq, n=4000))
            X = X.append(pd.DataFrame(fft_features).T)
            y = y.append([gen])
            
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)
    print(X.shape)
    return X, y

def model_training(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 5) 
     
    accs = []; depths = np.arange(60, 200, 20)
    for i in depths:
        print('training extra tree classifier, n estimators = {}'.format(i))
        etf = ExtraTreesClassifier(n_estimators = i, max_depth=None, min_samples_split=2, random_state=5).fit(X_train, y_train)
        print('accuracy {}'.format(round(etf.score(X_test, y_test),3)))
        accs += [etf.score(X_test, y_test)]
    
    print('top accuracy {}'.format(round(max(accs), 3)))
    dpth = depths[accs.index(max(accs))]
    etf = ExtraTreesClassifier(n_estimators = dpth, max_depth=None, min_samples_split=2, random_state=5).fit(X_train, y_train)
    
    joblib.dump(etf,'model_extratrees.pkl')
    print('model saved')


if __name__ == "__main__":
    X, y = fft_extracting()
    model_training(X, y)
    

