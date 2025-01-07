# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:57:48 2024

@author: cning
"""
# In[1]:
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython import display
import pickle

cm = 1/2.54

# In[2]:

import tensorflow.keras
print(tf.config.list_physical_devices())

runfile('load_data.py')

# In[3]:

GM_train = GM[ind_train,:,:]
STRUCT_train = Structure[ind_train,:]
RES_train = RES[ind_train,:,:]
GM_valid = GM[ind_valid,:]
STRUCT_valid = Structure[ind_valid,:]
RES_valid = RES[ind_valid,:]
GM_test = GM[ind_test,:,:]
STRUCT_test = Structure[ind_test,:]
RES_test = RES[ind_test,:,:]
print(GM_train.shape,RES_train.shape,GM_valid.shape,RES_valid.shape,GM_test.shape,RES_test.shape)

# In[4]:

def splitDataset(Structure,GM,RES,batchSize):
    n = Structure.shape[0] #number of samples
    ind = np.arange(n)
    np.random.shuffle(ind)
    Structure,GM,RES = Structure[ind,:],GM[ind,:,:],RES[ind,:,:]
    x1 = tf.data.Dataset.from_tensor_slices(Structure).batch(batchSize)
    x2 = tf.data.Dataset.from_tensor_slices(GM).batch(batchSize)
    y = tf.data.Dataset.from_tensor_slices(RES).batch(batchSize)
    return x1,x2,y

# In[transfer]

base_model = tf.keras.models.load_model(r'G:\research\11_surrogate_model2023_3\09_code\deterministic_SPR_Net\\best_base_model.h5')
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer.trainable = True
    else:
        layer.trainable = False
        
truncted_model = tf.keras.Model(inputs=base_model.input,
                                          outputs=base_model.layers[-4].output)

n_filters = 16
gm_inp = tf.keras.layers.Input(shape=(npoint, 2))
struct_inp = tf.keras.layers.Input(shape=(3,))
x0 = tf.keras.layers.Dense(15,activation='sigmoid')(struct_inp)
# Get the output of the -4th layer from the base model
base_out = truncted_model([x0, gm_inp])

struct_mid = tf.keras.layers.Dense(n_filters,activation='sigmoid')(struct_inp)
gm_mid = tf.keras.layers.Conv1D(n_filters,5,activation='tanh',padding='same')(gm_inp)
gm_mid = tf.keras.layers.Multiply()([gm_mid,struct_mid])
base_out = tf.keras.layers.Concatenate()([base_out,gm_mid])

pre_mu = tf.keras.layers.LSTM(n_filters,return_sequences=True)(base_out)
pre_mu  = tf.keras.layers.Conv1D(n_filters,5,activation='tanh',padding='same')(pre_mu) #batch,npoint,nfilters
mu  = tf.keras.layers.Conv1D(4,3,use_bias=False, activation='linear',padding='same')(pre_mu)
mu_model = tf.keras.Model([struct_inp,gm_inp],mu)

inp_sig = tf.keras.layers.Concatenate()([base_out,mu])
pre_sig = tf.keras.layers.Dense(n_filters,activation='relu')(inp_sig)
pre_sig = tf.keras.layers.Dense(n_filters,activation='relu')(pre_sig)
sig = tf.keras.layers.Dense(4, activation='softplus')(pre_sig)
sig_model = tf.keras.Model([struct_inp,gm_inp],sig)

model = tf.keras.Model([struct_inp,gm_inp], [mu,sig])

model.compile(metrics='mse')
model.summary()

# In[]:

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

optimizer1 = tf.keras.optimizers.Adam(1e-3)
optimizer2 = tf.keras.optimizers.Adam(1e-3)

def compute_loss(mu_model,sig_model, x1,x2,y):
    """"
    mean(predY),std(predY) = mu, sig
    log(|predY|) ~ normal(mu_x,sig_x)
    """
    mu = mu_model([x1,x2])
    sig = sig_model([x1,x2])
    angle = 2*tf.cast(mu>0,tf.float32)-1
    # mse =  tf.reduce_mean(tf.reduce_sum(tf.square(mu-y),axis=[1,2]))
    mu = tf.abs(mu)+1e-9 # mean for lognormal
    sig = tf.abs(sig)+1e-9
    mu_x = tf.math.log(mu**2/tf.math.sqrt(mu**2+sig**2))
    sig_x = tf.sqrt(tf.math.log(1+sig**2/mu**2))
    pred = tf.exp(mu_x)*angle
    # mse =  tf.reduce_mean(tf.reduce_sum(tf.square(pred-y),axis=[1,2]))
    mse =  tf.reduce_mean(tf.square(pred-y))
    
    square = tf.square(mu_x - tf.math.log(tf.abs(y)+1e-9))## preserve the same shape as y_pred.shape
    ms = 0.5*tf.divide(square,sig_x**2) + tf.math.log(sig_x)
    # ms = tf.reduce_mean(tf.reduce_sum(ms,axis=[1,2]))
    ms = tf.reduce_mean(ms)
    return mse,ms


def compute_apply_gradients(mu_model,sig_model,base_model, x1,x2,y, optimizer1, optimizer2):
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        mse,ms = compute_loss(mu_model,sig_model, x1,x2,y)
        mu_model.trainable=True
        gradients1 = tape1.gradient(mse, mu_model.trainable_variables)
        optimizer1.apply_gradients(zip(gradients1, mu_model.trainable_variables))
        mu_model.trainable=False
        gradients2 = tape2.gradient(ms, sig_model.trainable_variables)
        optimizer2.apply_gradients(zip(gradients2, sig_model.trainable_variables))
    return mse+ms
    
    
def plotPred(i0,model=model):
    x1,x2,y = Structure[i0:i0+1,:], GM[i0:i0+1,:,:], RES[i0:i0+1,:,:]
    mu,sig = model([x1,x2])
    angle = 2*tf.cast(y>0,tf.float32)-1
    mu_x = tf.math.log(mu**2/tf.math.sqrt(mu**2+sig**2))
    sig_x = tf.sqrt(tf.math.log(1+sig**2/mu**2))
    
    pred = destand(tf.exp(mu_x)*angle) #median prediction
    top = destand((tf.exp(mu_x+sig_x))*angle)
    bot = destand((tf.exp(mu_x-sig_x))*angle) 
    truey = destand(y)
    
    xt = np.arange(npoint)*dt
    plt.figure(figsize=(5,1))
    plt.plot(xt,truey[0,:,0],'r')
    plt.plot(xt,pred[0,:,0],'k',lw=0.8)
    plt.fill_between(xt, top[0,:,0], bot[0,:,0], color='gray', alpha=0.5)
    plt.xlim([0,60])
    plt.show()
    
    
# In[train]:


#create a write
epochs=1000
model.built = True
# model = tf.keras.models.load_model('best_model.h5')
df = pd.DataFrame()
losslist = []

start_time0 = time.time()
best_loss = 1e20
for epoch in range(1, epochs + 1):
    start_time = time.time()
    loss_train = []
    trainStruct,trainGM,trainRES = splitDataset(STRUCT_train,GM_train,RES_train,int(N_train/5))
    for (x1, x2, y) in zip(trainStruct,trainGM,trainRES):
        with tf.device('/GPU:0'):
            loss = compute_apply_gradients(mu_model,sig_model,base_model, x1,x2,y, optimizer1,optimizer2)
        loss_train.append(loss.numpy())
    end_time = time.time()
    loss_train = np.mean(loss_train)
    print(epoch, loss_train)
    
    if epoch % 5 == 0:
        loss_mu,loss_sig = compute_loss(mu_model,sig_model, STRUCT_valid,GM_valid,RES_valid)
        loss_valid = loss_mu+loss_sig

        losslist.append({'loss_train':loss_train,'mu_valid':loss_valid.numpy(),'sig_valid':loss_sig.numpy()})#append loss to list
        
        display.clear_output(wait=False)
        
        df = pd.DataFrame(losslist)  #append the dictionary to the dataframe
        df.to_csv('loss.csv')    #correct this

        print('Epoch: {},\n'
              'Validation set LOSS_sum: {:.2f},\n'
              'Training set LOSS_sum: {:.2f},\n'
              'time elapse for current epoch {:.2f},\n'
              'total time consumed {:.2f}'.format(epoch, loss_valid,loss_train, end_time - start_time, end_time - start_time0))
        i0 = np.random.randint(0,N)
        plotPred(i0)
        
        if loss_valid<best_loss:
            best_loss  = loss_valid
            model.save('best_model.h5')
    model.save('final_model.h5')
