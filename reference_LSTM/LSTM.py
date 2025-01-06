#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the needed Libraries

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# import sklearn

from IPython import display
import pickle
# import scipy

cm = 1/2.54

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

mpl.rcParams.update({"figure.figsize" : (9*cm,6*cm),
                 "figure.subplot.left" : 0.125, "figure.subplot.right" : 0.946,
                 "figure.subplot.bottom" : 0.125, "figure.subplot.top" : 0.965,
                 "axes.autolimit_mode" : "round_numbers",
                 "xtick.major.size"     : 5,
                 "xtick.minor.size"     : 3,
                 "xtick.major.width"    : 0.5,
                 "xtick.minor.width"    : 0.5,
                 "xtick.major.pad"      : 2,
                 "xtick.minor.visible" : True,
                 "ytick.major.size"     : 5,
                 "ytick.minor.size"     : 3,
                 "ytick.major.width"    : 0.5,
                 "ytick.minor.width"    : 0.5,
                 "ytick.major.pad"      : 2,
                 "ytick.minor.visible" : True,
                 "lines.markersize" : 5,
                 "lines.markerfacecolor" : "none",
                 "lines.markeredgewidth"  : 0.8})
plt.rcParams.update({'font.size': 8})


# In[2]:
print(tf.config.list_physical_devices())

# In[3]:

# exclude = [870,1324,1422,1448,1657,1723]
sf = 25;
dt = 2e-3*sf
N = 1950

# mask = np.ones(N, bool)
# mask[exclude] = False

open_file = open(r'G:\research\11_surrogate_model2023_3\00_data\ResponseLongTH.pkl', "rb")
temp = pickle.load(open_file) # L_record = [L_COL,L_ABU,L_BRD,L_COL_F,L_ABU_F,L_BRD_F]
open_file.close()
RESTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[3]],
         [ii[::sf] for ii in temp[2]],[ii[::sf] for ii in temp[5]]] #column, abutment,bearing
RESTH = np.array(RESTH).transpose((1,2,0))
# v = np.diff(d,axis=1)

open_file = open(r"G:\research\11_surrogate_model2023\00_extract_responses\data_AL2_differStruct\\GroundMotionTH.pkl", "rb")
temp = pickle.load(open_file) # STFT of GM [x,z]
open_file.close()
GMTH = [[ii[::sf] for ii in temp[0]],[ii[::sf] for ii in temp[1]]]
GMTH = np.array(GMTH).transpose((1,2,0))

npoint = GMTH.shape[1]
# N = GMTH.shape[0]
del temp


# In[4]:


StructureInfo = pd.read_excel(r'G:\research\11_surrogate_model2023_3\00_data\variables.xlsx')
# StructureInfo = StructureInfo.loc[list(mask),...]

Dir = StructureInfo.loc[:,'Dir'].values
StructureInfo = StructureInfo.iloc[:,:15]
FeatureName = StructureInfo.keys().tolist()
StructureInfo = StructureInfo.values
print(StructureInfo.shape,FeatureName)


# In[5]

fig = plt.figure(figsize=(5.4,1.))
gs = fig.add_gridspec(nrows=1,ncols=1,hspace=0.0,wspace=0.0)
axs = gs.subplots(sharex=True)
y=GMTH[630,:,0]*9.81
x=np.arange(len(y))*dt
axs.plot(x,y,lw=0.8,color='C3')
axs.grid(False)
axs.set_xlim([0,40])
axs.set_xlabel('t(s)',labelpad=2)
axs.set_ylabel('a(m/$s^2$)',labelpad=2)
axs.set_xticks(np.arange(0,45,5))
axs.margins(y=0.05*max(abs(y)))
axs.tick_params(axis="both",which='both',direction="in")
# axs.set_yticks(np.arange(-0.5,0.3,0.5))
plt.savefig("plot\\630GM.svg", dpi=100, transparent=True, bbox_inches='tight')
plt.show()


# In[5]:

# preprocessing ground motion STFT: (x-mu)/sigma
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
print(RES.shape)


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
    
# In[25]:


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


# In[26]:


def splitDataset(Structure,GM,RES,batchSize):
    n = Structure.shape[0] #number of samples
    ind = np.arange(n)
    np.random.shuffle(ind)
    Structure,GM,RES = Structure[ind,:],GM[ind,:,:],RES[ind,:,:]
    x1 = tf.data.Dataset.from_tensor_slices(Structure).batch(batchSize)
    x2 = tf.data.Dataset.from_tensor_slices(GM).batch(batchSize)
    y = tf.data.Dataset.from_tensor_slices(RES).batch(batchSize)
    return x1,x2,y


# In[31]:
# hyper-parameters
n_layers = 12
n_filters = 32
units = 4
filter_width = 2

# define an input history series and pass it through a stack of dilated causal convolution blocks
gm_inp = tf.keras.layers.Input(shape=(npoint, 2))
x = gm_inp
for ii in range(n_layers):
    x = tf.keras.layers.LSTM(n_filters,return_sequences=True)(x)
out = tf.keras.layers.Conv1D(n_filters, filter_width, padding='same')(x)
out = tf.keras.layers.Conv1D(4, 1, padding='same')(out)  # batch, npoint, 3
train_res = tf.keras.layers.Dropout(.3)(out)

model = tf.keras.Model(gm_inp, train_res)
model.summary()

# In[13]:


optimizer = tf.keras.optimizers.Adam(1e-3)

#@tf.function
def compute_loss(model, x,y):
    y_logit = model(x)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_logit),axis=[1,2]))
    return loss

#@tf.function
def compute_apply_gradients(model, x,y, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x,y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def plotPred(i0,model=model):
    x,y =  GM[i0:i0+1,:,:], RES[i0:i0+1,:,:]
    pred = model(x)

    pred = destand(pred) #median prediction
    truey = destand(y)
    
    xt = np.arange(npoint)*dt
    plt.figure(figsize=(5,1))
    plt.plot(xt,truey[0,:,0],'r')
    plt.plot(xt,pred[0,:,0],'k',lw=0.8)
    plt.xlim([0,60])
    plt.show()

# In[13]:


#create a write
epochs=3000
model.built = True
# model.load_weights('base_model_weights.h5')
# model = tf.keras.models.load_model('base_model.h5')
df = pd.DataFrame()
losslist = []

start_time0 = time.time()
best_loss = 1e20
for epoch in range(1, epochs + 1):
    start_time = time.time()
    loss_train,loss_regul,loss_recon = [], [], []
    trainStruct,trainGM,trainRES = splitDataset(STRUCT_train,GM_train,RES_train,int(N_train/5))
    for (x, y) in zip(trainGM,trainRES):
        with tf.device('/GPU:0'):
            loss = compute_apply_gradients(model, x,y, optimizer)
        loss_train.append(loss.numpy())
    end_time = time.time()
    loss_train = np.mean(loss_train)
    print(epoch)
    
    if epoch % 5 == 0:
        loss_valid = compute_loss(model, GM_valid,RES_valid)
        losslist.append({'loss_train':loss_train,'loss_valid':loss_valid.numpy()})#append loss to list
        
        display.clear_output(wait=False)
        
        df = pd.DataFrame(losslist)  #append the dictionary to the dataframe
        df.to_csv('loss.csv')    #correct this

        print('Epoch: {},\n'
              'Training set Loss: {:.2f},\n'
              'Validation set LOSS_sum: {:.2f},\n'
              'time elapse for current epoch {:.2f},\n'
              'total time consumed {:.2f}'.format(epoch,loss_train, loss_valid, end_time - start_time, end_time - start_time0))
        i0 = np.random.randint(0,N)
        plotPred(i0)
        
        if loss_valid<best_loss:
            best_loss  = loss_valid
            model.save_weights('base_model_weights.h5')
    model.save_weights('base_weights.h5') #prefered loss ~==30

# In[14]:


df = pd.read_csv('loss.csv',encoding = 'unicode_escape')
#df = df[5:]
fig = plt.figure(figsize=(2.5,2))
gs = fig.add_gridspec(1)
axs = gs.subplots()
axs.plot(np.arange(0,df.shape[0])*5,df['Training'],label='Train',color='red',lw=0.8)
axs.plot(np.arange(0,df.shape[0])*5,df['Validation'],label='Validation',color='blue',lw=0.8)
axs.set_xlabel('Epoch')
axs.set_ylabel('Loss')
axs.legend()
axs.set_ylim(0,5500)
axs.set_xlim(0,1000)
plt.savefig("plot\loss_epoch.svg", dpi=100, transparent=True, bbox_inches='tight')
plt.show()


# In[15]:


print(min(df['loss_valid']))
print(min(df['loss_train']))

# # Reconstruct time history

# In[32]:

model.built = True
model.load_weights('base_model_weights.h5')

with tf.device('/CPU:0'):
    predRES = model(GM)
predRES = np.array(predRES)

predRES = destand(predRES)
RES = RESTH.astype('float32')


# In[33]

def compare_plot(ind, path,dt = dt):
    
    fig = plt.figure(figsize=(6*cm,8*cm))
    gs = fig.add_gridspec(nrows=4,ncols=1,hspace=0.2,wspace=0.0,height_ratios=[1,1,1,1])
    axs = gs.subplots(sharex=True)
    [ax.tick_params(axis="both",which='both',direction="in") for ax in axs]
    
    tmax = 30
    l = int(tmax/dt)
    x0 = np.arange(0,l)*dt
    
    axs[0].plot(x0,100*RES[ind,:l,0],linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[0].plot(x0,100*predRES[ind,:l,0],linewidth=0.5,linestyle='-',color='r',label='Predicted')
    
    scale=0.004448
    axs[1].plot(x0,RES[ind,:l,1]*scale,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[1].plot(x0,predRES[ind,:l,1]*scale,linewidth=0.5,linestyle='-',color='r',label='Predicted')

    scale=2.54
    axs[2].plot(x0,RES[ind,:l,2]*scale,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[2].plot(x0,predRES[ind,:l,2]*scale,linewidth=0.5,linestyle='-',color='r',label='Predicted')
    
    scale=0.004448
    axs[3].plot(x0,RES[ind,:l,3]*scale,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[3].plot(x0,predRES[ind,:l,3]*scale,linewidth=0.5,linestyle='-',color='r',label='Predicted')

    axs[0].set_ylabel('$\Delta_c(\%)$',labelpad=2)
    axs[1].set_ylabel("$F_c (MN)$",labelpad=2)
    axs[2].set_ylabel('$\gamma (cm)$',labelpad=2)
    axs[3].set_ylabel('$F_b (MN)$',labelpad=2)
    axs[3].set_xlabel('$t(s)$',labelpad=4)
    axs[3].set_xticks(np.arange(0,65,5))
    axs[3].set_xlim(0,tmax)
    #axs[2].legend(loc='lower right',fontsize=7)

    axs[0].tick_params(labelbottom=False);axs[1].tick_params(labelbottom=False);axs[2].tick_params(labelbottom=False);
    #ax3.set_ylim(1.1*min(RES_record[0][ind]),1.1*max(RES_record[0][ind]))
    #axs[0].set_yticks(np.arange(-2.,2.5,1))
    #axs[0].set_ylim(-2.5,2.5)
    #axs[1].set_yticks(np.arange(-0.02,0.03,0.01))
    #axs[1].set_ylim(-.025,0.025)
    #axs[2].set_yticks(np.arange(-10,15,5))
    #axs[2].set_ylim(-15,15)

    # plt.suptitle('gm#{}'.format(ind))
    # plt.savefig(path+"comparison #"+str(ind+1)+".png", dpi=100, transparent=True, bbox_inches='tight')
    #plt.savefig("plot\comparison #"+str(ind+1)+".pdf", dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(path+str(ind+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()

for ii in np.argsort(np.max(np.abs(RES[ind_test,:,0]),axis=1))[[130,403,678]]:
    print(np.max(np.abs(RES[ind_test[ii],:,0])),np.max(np.abs(predRES[ind_test[ii],:,0])))
    compare_plot(ind_test[ii],path="plot\\testing\\")


# In[20]:


for ii in ind_train[0:10]:
    compare_plot(ii,path="plot\\training\\")
# In[]

for ii in ind_valid[0:10]:
    compare_plot(ii,path="plot\\validating\\")

# In[testining]

for ii in [75,1135,1920]:
    compare_plot(ii,path="plot\\testing\\")
    

# In[compare]
from metrics2 import Cal_cdf

xlabel = [r"$\Delta_c$ (%)", r"$F_c (MN)$",r"$\Delta_b (mm)$",r"$F_b (kN)$"]
scale  = [100, 0.004448, 25.4, 4.448]
xlim = [[0.05,3],[0.5,10],[5,150],[10,500]]

for ii in np.arange(4):
    true_test = RES[ind_test,:,ii]*scale[ii]
    pred_median = predRES[ind_test,:,ii]*scale[ii]

    ind = np.argmax(np.abs(true_test),axis=1).tolist()
    cr = np.max(np.abs(true_test),axis=1)
    
    # c0 = np.abs(pred_median)[np.arange(len(ind)),ind]
    c0 = np.max(np.abs(pred_median),axis=1)
    
    plt.figure(figsize=(2.5,2.2))
    x,y = Cal_cdf(cr)
    plt.plot(x,y,'k-',label='truth',lw=1.2)

    x0,y0 = Cal_cdf(c0)
    plt.plot(x0,y0,'--r',label='median',lw=0.8)

    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylim([0,1])
    plt.ylabel('CDF',labelpad=2)
    plt.grid(which='major',axis='y',linewidth=0.5)
    plt.grid(which='both',axis='x',linewidth=0.3)
    plt.xlim(xlim[ii])
    # plt.legend()
    plt.tight_layout()
    # plt.savefig("plot\\"+xlabel[ii][1:-1]+".svg",dpi=100)
    plt.show()
    
# In[energy]
StructureInfo = pd.read_excel(r'G:\research\11_surrogate_model2023_3\00_data\variables.xlsx').iloc[:,:16]
Hc = StructureInfo['H(m)'].values[:,None]

trueEC = np.trapz(RESTH[:,:,1]*4.448,-RESTH[:,:,0]*Hc) #kN*m
trueEB = np.trapz(RESTH[:,:,3]*4.448,-RESTH[:,:,2]*0.0254) #kN*m
truePC = np.max(np.abs(RESTH[:,:,0]),axis=1)
truePB = np.max(np.abs(RESTH[:,:,2]),axis=1)

medEC = np.trapz(predRES[:,:,1]*4.448,-predRES[:,:,0]*Hc)
medEB = np.trapz(predRES[:,:,3]*4.448,-predRES[:,:,2]*0.0254)

xlabel = [r"$E_c (kN\bullet m)$", r"$E_b (kN\bullet m)$"]
xlim = [[1,1e4],[100,1e3]]
xticks = [[1,10,1e2,1e3,1e4],[0.1,1,10,1e2,1e3]]


for ii in np.arange(len(xlabel)):
    if ii==0: 
        ind_temp=ind_test
    else:
        ind_temp = list(set(np.where(trueEB>0.1)[0]).intersection(set(ind_test)))
        # ind_temp = list(set(np.where(medEB>100)[0]).intersection(set(ind_test)))
        # ind_temp = ind_test
        
    true_test = [trueEC[ind_temp],trueEB[ind_temp]][ii]
    pred_median = [medEC[ind_temp],medEB[ind_temp]][ii]
    
    plt.figure(figsize=(2.5,2.2))
    x,y = Cal_cdf(true_test)
    plt.plot(x,y,'k-',label='truth',lw=1.2)
    
    x0,y0 = Cal_cdf(pred_median)
    plt.plot(x0,y0,'--r',label='median',lw=0.8)

    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylim([0,1])
    plt.ylabel('CDF',labelpad=2)
    plt.grid(which='major',axis='y',linewidth=0.5)
    plt.grid(which='major',axis='x',linewidth=0.3)
    plt.xlim(xlim[ii])
    plt.xticks(xticks[ii])
    plt.minorticks_off()
    # plt.legend()
    plt.tight_layout()
    # plt.savefig("plot\\"+xlabel[ii][1:4]+".svg",dpi=100)
    plt.show()
# In[]
for ii in [212,130,630]:
    force,disp = RESTH[ii,:,1]*4.448,-RESTH[ii,:,0]*Hc[ii]
    predF,predD = predRES[ii,:,1]*4.448, -predRES[ii,:,0]*Hc[ii]
    realE = [np.trapz(force[0:jj],disp[0:jj]) for jj in np.arange(len(force))]
    predE = [np.trapz(predF[0:jj],predD[0:jj]) for jj in np.arange(len(predF))]
    
    plt.figure(figsize=(2.2,1.8))
    plt.plot(np.arange(npoint)*dt,realE,'k',lw=1.2)
    plt.plot(np.arange(npoint)*dt,predE,'r-',lw=0.8)
    plt.yscale('log')
    plt.xlabel('t(s)',labelpad=2)
    plt.ylabel(r'$kN\bullet m$',labelpad=2)
    plt.ylim(1e-2,)
    plt.xlim([0,40])
    # plt.savefig("plot\\\Ec_"+str(ii+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(2.2,1.8))
    plt.plot(-RESTH[ii,:,0]*100,RESTH[ii,:,1]*0.004448,'k',lw=1.2,label='truth')
    plt.plot(-predRES[ii,:,0]*100,predRES[ii,:,1]*0.004448,'r',lw=0.8,label='meidin')
    # plt.plot(-predLow[ii,:,0]*100,predLow[ii,:,1]*0.004448,'gray',lw=0.8,alpha=0.5,label='$\sigma$')
    # plt.plot(-predUp[ii,:,0]*100,predUp[ii,:,1]*0.004448,'gray',lw=0.8,alpha=0.5,label='$\sigma$')
    plt.xlabel('$\Delta_c $ (%)')
    plt.ylabel('$F_c$ (MN)')
    # plt.savefig("plot\\\\Hystersis_"+str(ii+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
    
    force,disp = RESTH[ii,:,3]*4.448,-RESTH[ii,:,2]*0.0254
    predF,predD = predRES[ii,:,3]*4.448, -predRES[ii,:,2]*0.0254
    realE = [np.trapz(force[0:jj],disp[0:jj]) for jj in np.arange(len(force))]
    predE = [np.trapz(predF[0:jj],predD[0:jj]) for jj in np.arange(len(predF))]
    plt.plot(np.arange(npoint)*dt,realE,'k',lw=1.2)
    plt.plot(np.arange(npoint)*dt,predE,'r',lw=0.8)
    # plt.yscale('log')
    # plt.ylim(1e-3,)
    plt.xlim([0,40])
    plt.xlabel('t(s)',labelpad=2)
    plt.ylabel(r'$kN\bullet m$')
    plt.show()
    
    plt.figure(figsize=(2.5,2.2))
    plt.plot(-RESTH[ii,:,2]*2.54,RESTH[ii,:,3],'k',lw=1.5,label='truth')
    plt.plot(-predRES[ii,:,2]*2.54,predRES[ii,:,3],'r',lw=0.8,label='meidin')
    # plt.plot(-predLow[ii,:,0]*100,predLow[ii,:,1],'k',lw=0.8,alpha=0.5,label='$\sigma$')
    # plt.plot(-predUp[ii,:,0]*100,predUp[ii,:,1],'k',lw=0.8,alpha=0.5,label='$\sigma$')
    plt.xlabel('$\Delta_b $ (cm)')
    plt.ylabel('$F_b$ (MN)')
    plt.show()
    
    compare_plot(ii,path="plot\\testing\\")

# In[]
StructureInfo = pd.read_excel(r'G:\research\11_surrogate_model2023_3\00_data\variables.xlsx').iloc[:,:16]
Hc = StructureInfo['H(m)'].values[:,None]

trueEC = np.trapz(RESTH[:,:,1]*4.448,-RESTH[:,:,0]*Hc) #kN*m
trueEB = np.trapz(RESTH[:,:,3]*4.448,-RESTH[:,:,2]*0.0254) #kN*m
truePC = np.max(np.abs(RESTH[:,:,0]),axis=1)*100
truePB = np.max(np.abs(RESTH[:,:,2]),axis=1)*2.54 #cm

medEC = np.trapz(predRES[:,:,1]*4.448,-predRES[:,:,0]*Hc)
medEB = np.trapz(predRES[:,:,3]*4.448,-predRES[:,:,2]*0.0254)
medPC = np.max(np.abs(predRES[:,:,0]),axis=1)*100
medPB = np.max(np.abs(predRES[:,:,2]),axis=1)*2.54

# In[]
import scipy

fig = plt.figure(figsize=(2.5,2.2))
plt.scatter(truePC[ind_train],medPC[ind_train],color='green',s=2,edgecolor='k',lw=0.3,label='training')
plt.scatter(truePC[ind_test],medPC[ind_test],color='orange',s=2,edgecolor='k',lw=0.3,label='testing',alpha=0.8)
plt.plot([0,8],[0,8],'r--',lw=1.2)
plt.xlabel('ground truth $\Delta$(%)',labelpad=2)
plt.ylabel('prediction $\Delta$(%)',labelpad=2)
plt.xticks(np.arange(0,8))
plt.xlim([0,4])
plt.ylim([0,4])
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc=4,fontsize=8)
plt.tight_layout()
# plt.savefig('plot\\drift_compare.svg',dpi=100)
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(truePC[ind_test],medPC[ind_test])
print(slope, intercept, r_value, p_value, std_err)


fig = plt.figure(figsize=(2.4,2.2))
plt.scatter(trueEC[ind_train]/1e3,medEC[ind_train]/1e3,color='green',s=2,edgecolor='k',lw=0.3,label='training')
plt.scatter(trueEC[ind_test]/1e3,medEC[ind_test]/1e3,color='orange',s=2,edgecolor='k',lw=0.3,label='testing',alpha=0.8)
plt.plot([0,6],[0,6],'r--',lw=1.2)
plt.xlabel(r'ground truth $E_c (MN\bullet m)$',labelpad=2)
plt.ylabel(r'prediction $E_c (MN\bullet m)$',labelpad=2)
# plt.xticks(np.arange(0,8))
plt.xlim([0,6])
plt.ylim([0,6])
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc=4,fontsize=8)
plt.tight_layout()
# plt.savefig('plot\\Ec_compare.svg',dpi=100)
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(trueEC[ind_test]/1e3,medEC[ind_test]/1e3)
print(slope, intercept, r_value, p_value, std_err)