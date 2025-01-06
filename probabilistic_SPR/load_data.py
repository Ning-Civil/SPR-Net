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

# mask = np.ones(N, bool)
# mask[exclude] = False

open_file = open(r'G:\research\11_surrogate_model2023_3\09_code\data\\ResponseLongTH.pkl', "rb")
temp = pickle.load(open_file) 
open_file.close()
RESTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[3]],
         [ii[::sf] for ii in temp[2]],[ii[::sf] for ii in temp[5]]] #column, abutment,bearing
RESTH = np.array(RESTH).transpose((1,2,0))

open_file = open(r"G:\research\11_surrogate_model2023_3\09_code\data\\GroundMotionTH.pkl", "rb")
temp = pickle.load(open_file) # STFT of GM [x,z]
open_file.close()
GMTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[1]]]
GMTH = np.array(GMTH).transpose((1,2,0))

npoint = GMTH.shape[1]
# N = GMTH.shape[0]
del temp

# In[4]:

StructureInfo = pd.read_excel(r'G:\research\11_surrogate_model2023_3\09_code\data\\variables.xlsx').iloc[:,:16]
# StructureInfo = StructureInfo.loc[list(mask),...]
FeatureName = ["L(m)","lambda","abut_gap(mm)"]
Dir = StructureInfo.loc[:,'Dir'].values

StructureInfo = StructureInfo.loc[:,FeatureName]
StructureInfo = StructureInfo.values
print(StructureInfo.shape,FeatureName)

# In[5]:


GM = np.zeros((N,npoint,2)).astype('float32')
for ii in np.arange(N).astype('int'):
    GM[ii,:,0]=GMTH[ii,:,Dir[ii]-2]
    GM[ii,:,1]=GMTH[ii,:,Dir[ii]-1]

GM = (GM-GM.mean())/GM.std();

meanStructure,stdStructure = StructureInfo.mean(axis=0),StructureInfo.std(axis=0)
Structure = (StructureInfo-meanStructure)/stdStructure

print(GM.shape,Structure.shape)


# In[23]:

# RES = np.diff(RESTH,axis=1,prepend=0).astype('float32')
RES = RESTH.astype('float32')

def stand(RES):
    # RES.shape=(N,npoint,3)
    RES_mean = []
    RES_std = []
    for ii in np.arange(4):
        # RES_mean.append(RES[:,:,ii].mean())
        RES_mean.append(0)
        RES_std.append(RES[:,:,ii].std())
        RES[:,:,ii] = (RES[:,:,ii]-RES_mean[ii])/(RES_std[ii])
    return RES,RES_mean,RES_std

RES,RES_mean,RES_std = stand(RES)

def destand(res,RES_mean=RES_mean,RES_std=RES_std):
    # res.shape = (batch,npoint,3)
    if len(res.shape)<3: # res.shape=(npoint,3)
        res = res[None,...]
    out = np.zeros(res.shape)
    for ii in range(4):
        out[:,:,ii] = res[:,:,ii]*RES_std[ii]+RES_mean[ii]
    return out


# In[24]:
r = np.random.RandomState(1)
ind = np.arange(int(N/3))
r.shuffle(ind)

N_train = int(900/3)
N_valid = int(240/3)
N_test = int(len(ind)-N_train-N_valid)

ind_train = np.concatenate((ind[0:N_train],int(N/3)+ind[0:N_train],int(2*N/3)+ind[0:N_train])).tolist()
ind_valid = np.concatenate((ind[N_train:N_train+N_valid],int(N/3)+ind[N_train:N_train+N_valid],int(2*N/3)+ind[N_train:N_train+N_valid])).tolist()
ind_test = np.concatenate((ind[N_train+N_valid:],int(N/3)+ind[N_train+N_valid:],int(2*N/3)+ind[N_train+N_valid:])).tolist()
