# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:57:48 2024

@author: cning
"""
# In[1]:
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import pickle

# In[3]:
sf = 25;
dt = 2e-3*sf
N = 1950

open_file = open(r'G:\research\11_surrogate_model2023_3\02_specific_structure\01_extract_data\data\\soft_ResponseLongTH.pkl', "rb")
temp = pickle.load(open_file) 
open_file.close()
soft_RESTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[3]],
          [ii[::sf] for ii in temp[2]],[ii[::sf] for ii in temp[5]]] #column, abutment,bearing
soft_RESTH = np.array(soft_RESTH).transpose((1,2,0))
soft_RESTH = soft_RESTH*np.array([100,0.004448,2.54,0.004448]).T
del temp

open_file = open(r'G:\research\11_surrogate_model2023_3\02_specific_structure\01_extract_data\data\\median_ResponseLongTH.pkl', "rb")
temp = pickle.load(open_file) 
open_file.close()
median_RESTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[3]],
          [ii[::sf] for ii in temp[2]],[ii[::sf] for ii in temp[5]]] #column, abutment,bearing
median_RESTH = np.array(median_RESTH).transpose((1,2,0))
median_RESTH = median_RESTH*np.array([100,0.004448,2.54,0.004448]).T
del temp

open_file = open(r'G:\research\11_surrogate_model2023_3\02_specific_structure\01_extract_data\data\\stiff_ResponseLongTH.pkl', "rb")
temp = pickle.load(open_file) 
open_file.close()
stiff_RESTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[3]],
          [ii[::sf] for ii in temp[2]],[ii[::sf] for ii in temp[5]]] #column, abutment,bearing
stiff_RESTH = np.array(stiff_RESTH).transpose((1,2,0))
stiff_RESTH = stiff_RESTH*np.array([100,0.004448,2.54,0.004448]).T
del temp

open_file = open(r"G:\research\11_surrogate_model2023_3\09_code\data\\GroundMotionTH.pkl", "rb")
temp = pickle.load(open_file) # STFT of GM [x,z]
open_file.close()
GMTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[1]]]
GMTH = np.array(GMTH).transpose((1,2,0))
npoint = GMTH.shape[1]

del temp

PGA = pd.read_excel(r"G:/research/11_surrogate_model2023_3/09_code/data/variables.xlsx")['PGA'].values

Dir = np.loadtxt(r'G:/research/11_surrogate_model2023_3/02_specific_structure/01_extract_data/direction.txt').astype('int')
npoint = soft_RESTH.shape[1]
N = soft_RESTH.shape[0]

# In[standardize]
GM_mean,GM_std = 2.6309976e-06, 0.05746545
GM = np.zeros((N,npoint,2)).astype('float32')
for ii in np.arange(N).astype('int'):
    GM[ii,:,0]=GMTH[ii,:,Dir[ii]-2]
    GM[ii,:,1]=GMTH[ii,:,Dir[ii]-1]

GM = (GM-GM_mean)/GM_std;

meanStructure = np.array([31.36269298,  4.66389415, 23.30560718])
stdStructure = np.array([ 4.80753529,  0.90719497, 10.64620687])

RES_mean = [0,0,0,0]
RES_std = [0.002313021, 128.84035, 0.74727243, 8.932909]

def destand(res,RES_mean=RES_mean,RES_std=RES_std):
    # res.shape = (batch,npoint,3)
    if len(res.shape)<3: # res.shape=(npoint,3)
        res = res[None,...]
    out = np.zeros(res.shape)
    scale = [100,0.004448,2.54,0.004448] #percent,MN,cm,MN
    for ii in range(4):
        out[:,:,ii] = (res[:,:,ii]*RES_std[ii]+RES_mean[ii])*scale[ii]
    return out
