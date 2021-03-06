#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rozpoznawanie ganunku nowych utworów, opcjonalnie kopiowanie do odpowiedniego folderu
python3 genres_recogn.py utwor.wav #rozpoznaje tylko podany utwór
python3 genres_recogn.py #rozpoznaje wszystkie pliki *.wav w bieżącej lokalizacji
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import imp
import joblib
import scipy.io.wavfile
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def genre(file):
        _, music_freq = scipy.io.wavfile.read(file)
        X_pred = abs(np.fft.rfft(music_freq, n=4000))
        clf = joblib.load('model_extratrees.pkl')
        gen = clf.predict(X_pred.reshape(1, -1))[0]
        print(file, gen)
        #os.rename(file, os.getcwd()+'/genres/'+gen+'/'+file)
        
def files():
    music_files = [e for e in os.listdir('.') if '.wav' in e]
    for e in music_files:
        genre(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        genre(sys.argv[1])
    else: 
        files()


