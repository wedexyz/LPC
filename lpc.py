import numpy as np # linear algebra
import pandas as pd 
import os
import librosa
from numpy import *

'''path = 'A1.wav'
data, sampling_rate = librosa.load(path)
lpcs = librosa.lpc(data, 16)
print(lpcs)'''

def autocorr(self, order=None):
    if order is None:
        order = len(self) - 1
    return [sum(self[n] * self[n + tau] for n in range(len(self) - tau)) for tau in range(order + 1)]

def core_lpcc(seq, err_term, order=None):
    if order is None:
        order = len(seq) - 1
    lpcc_coeffs = [np.log(err_term), -seq[0]]
    for n in range(2, order + 1):
        # Use order + 1 as upper bound for the last iteration
        upbound = (order + 1 if n > order else n)
        lpcc_coef = -sum(i * lpcc_coeffs[i] * seq[n - i - 1]
                         for i in range(1, upbound)) * 1. / upbound
        lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
        lpcc_coeffs.append(lpcc_coef)
    return lpcc_coeffs

def lpcc(lpcorder=None, cepsorder=None):
        coefs =  librosa.lpc( lpcorder,cepsorder)
        acseq =  np.array(autocorr(lpcorder, cepsorder))
        err_term =np.sqrt( acseq[0] + sum(a * c for a, c in zip(acseq[1:], coefs)))
        return core_lpcc(coefs, err_term, cepsorder)

'''lpccs = lpcc(data,16)
print(lpccs)


point1 = np.array(lpcs)
point2 = np.array(lpccs)


dist = np.linalg.norm(point1 - point2)


print(dist)'''
