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
import cmcrameri.cm as cmc

from IPython import display
import pickle
 
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

from tensorflow.keras.regularizers import l2, l1

# hyper-parameters
n_filters = 16
units = 4
filter_width = 2
dilation_rates = [2**i for i in range(12)]
factor = 0.

# define an input history series and pass it through a stack of dilated causal convolution blocks
gm_inp = tf.keras.layers.Input(shape=(npoint, 2))
struct_inp = tf.keras.layers.Input(shape=(15,))
x = gm_inp
x0 = struct_inp

skips = []
for dilation_rate in dilation_rates:
    x0 = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=l1(0))(x0)
    x0 = tf.keras.layers.Dense(n_filters, activation='sigmoid', kernel_regularizer=l1(0))(x0)

    # preprocessing - equivalent to time-distributed dense
    x = tf.keras.layers.Conv1D(n_filters, 1, padding='same', kernel_regularizer=l1(factor))(x)

    # filter
    x_f = tf.keras.layers.Conv1D(filters=n_filters,
                                 kernel_size=filter_width,
                                 padding='causal',
                                 dilation_rate=dilation_rate, kernel_regularizer=l1(factor))(x)  # activation=None

    # gate
    x_g = tf.keras.layers.Conv1D(filters=n_filters,
                                 kernel_size=filter_width,
                                 padding='causal',
                                 dilation_rate=dilation_rate, kernel_regularizer=l1(factor))(x)

    # combine filter and gating branches
    z = tf.keras.layers.Multiply()([tf.keras.layers.Activation('tanh')(x_f),
                                    tf.keras.layers.Activation('sigmoid')(x_g),x0])

    # postprocessing - equivalent to time-distributed dense
    z = tf.keras.layers.Conv1D(n_filters, 1, padding='same', activation='linear', kernel_regularizer=l1(factor))(z)

    # residual connection
    x = tf.keras.layers.Add()([x, z])

    # collect skip connections
    skips.append(z)

# add all skip connection outputs
out = tf.keras.layers.Activation('tanh')(tf.keras.layers.Add()(skips))

out = tf.keras.layers.LSTM(n_filters, return_sequences=True)(out)
out = tf.keras.layers.Conv1D(n_filters, filter_width, padding='same', kernel_regularizer=l1(factor))(out)
out = tf.keras.layers.Conv1D(4, 1, padding='same', kernel_regularizer=l1(factor))(out)  # batch, npoint, 3
train_res = tf.keras.layers.Dropout(.3)(out)

model = tf.keras.Model([struct_inp, gm_inp], train_res)

# In[13]:


optimizer = tf.keras.optimizers.Adam(1e-3)

model.compile(metrics='mse')
model.summary()


#@tf.function
def compute_loss(model, x1,x2,y):
    y_logit = model([x1,x2])
    regul_loss = sum(model.losses)
    recon_1 = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_logit),axis=[1,2]))
    # recon_2 = tf.reduce_mean(tf.square(tf.reduce_sum(tf.abs(y),axis=[1,2])-tf.reduce_sum(tf.abs(y_logit),axis=[1,2])))
    recon_loss = recon_1
    return regul_loss+recon_loss, recon_loss, regul_loss

#@tf.function
def compute_apply_gradients(model, x1,x2,y, optimizer):
    with tf.GradientTape() as tape:
        loss,recon,regul = compute_loss(model, x1,x2,y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss,recon,regul

def plotPred(i0,model=model):
    x1,x2,y =  Structure[i0:i0+1,:],GM[i0:i0+1,:,:], RES[i0:i0+1,:,:]
    pred = model([x1,x2])

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
epochs=1000
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
    for (x1, x2, y) in zip(trainStruct,trainGM,trainRES):
        with tf.device('/GPU:0'):
            loss,recon,regul = compute_apply_gradients(model, x1,x2,y, optimizer)
        loss_train.append(loss.numpy())
        loss_regul.append(regul.numpy())
        loss_recon.append(recon.numpy())
    end_time = time.time()
    loss_train = np.mean(loss_train)
    loss_regul = np.mean(loss_regul)
    loss_recon = np.mean(loss_recon)
    print(epoch)
    
    if epoch % 5 == 0:
        loss_valid,regul,recon = compute_loss(model, STRUCT_valid,GM_valid,RES_valid)
        losslist.append({'loss_train':loss_train,'loss_regul':loss_regul,'loss_recon':loss_recon,'loss_valid':loss_valid.numpy()})#append loss to list
        
        display.clear_output(wait=False)
        
        df = pd.DataFrame(losslist)  #append the dictionary to the dataframe
        df.to_csv('loss.csv')    #correct this

        print('Epoch: {},\n'
              'Validation set LOSS_sum: {:.2f},\n'
              'Training regularization: {:.2f},\n'
              'Training reconstruction: {:.2f},\n'
              'time elapse for current epoch {:.2f},\n'
              'total time consumed {:.2f}'.format(epoch, loss_valid,loss_regul, loss_recon, end_time - start_time, end_time - start_time0))
        i0 = np.random.randint(0,N)
        plotPred(i0)
        
        if loss_valid<best_loss:
            best_loss  = loss_valid
            model.save('best_base_model.h5')
    model.save('final_base_model.h5') #prefered loss ~==30

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
# axs.legend()
axs.set_ylim(0,5500)
axs.set_xlim(0,1000)
plt.savefig("plot\loss_epoch.svg", dpi=100, transparent=True, bbox_inches='tight')
plt.show()


# In[15]:


print(min(df['loss_valid']))
print(min(df['loss_train']))

# # Reconstruct time history

# In[32]:


# model.built = True
# model.load_weights('base_weights.h5')
# model.load_weights('base_model_weights.h5')
model = tf.keras.models.load_model('best_base_model.h5')
with tf.device('/CPU:0'):
    predRES = model([Structure,GM])
predRES = np.array(predRES)

predRES = destand(predRES)
RES = RESTH.astype('float32')


 # In[33]:
# predRES = np.cumsum(predRES,axis=1)
# RES = np.cumsum(RES,axis=1)

# from matplotlib.ticker import FormatStrFormatter
# from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare_plot(ind, path,dt = dt):
    
    fig = plt.figure(figsize=(6*cm,8*cm))
    gs = fig.add_gridspec(nrows=4,ncols=1,hspace=0.2,wspace=0.0,height_ratios=[1,1,1,1])
    axs = gs.subplots(sharex=True)

    axs[0].plot(np.arange(0,npoint)*dt,100*RES[ind,:,0],linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[0].plot(np.arange(0,npoint)*dt,100*predRES[ind,:,0],linewidth=0.5,linestyle='solid',color='r',label='Predicted')
    axs[0].grid()
    
    axs[1].plot(np.arange(0,npoint)*dt,RES[ind,:,1]*0.0044482,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[1].plot(np.arange(0,npoint)*dt,predRES[ind,:,1]*0.004482,linewidth=0.5,linestyle='solid',color='r',label='Predicted')
    axs[1].grid()
    
    axs[2].plot(np.arange(0,npoint)*dt,RES[ind,:,2]*25.4,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[2].plot(np.arange(0,npoint)*dt,predRES[ind,:,2]*25.4,linewidth=0.5,linestyle='solid',color='r',label='Predicted')
    axs[2].grid()
    
    axs[3].plot(np.arange(0,npoint)*dt,RES[ind,:,3]*4.4482,linewidth=1.,color='k',label='Ground truth',alpha=0.8)
    axs[3].plot(np.arange(0,npoint)*dt,predRES[ind,:,3]*4.4482,linewidth=0.5,linestyle='solid',color='r',label='Predicted')
    axs[3].grid()
        
    # print(np.abs(RES[ind,:,1]).max(),np.abs([predRES[ind,:,1]]).max())
    axs[0].set_ylabel('$\Delta_c(\%)$')
    axs[1].set_ylabel("$F_c (MN)$")
    axs[2].set_ylabel('$\gamma (mm)$')
    axs[3].set_ylabel('$F_n (kN)$')
    axs[3].set_xlabel('$t(s)$')
    axs[3].set_xticks(np.arange(0,45,5))
    axs[3].set_xlim(0,30)
    axs[0].grid(False);axs[1].grid(False);axs[2].grid(False);axs[3].grid(False)
    #axs[2].legend(loc='lower right',fontsize=7)

    axs[0].tick_params(labelbottom=False);axs[1].tick_params(labelbottom=False)
    axs[2].tick_params(labelbottom=False);
    #ax3.set_ylim(1.1*min(RES_record[0][ind]),1.1*max(RES_record[0][ind]))
    #axs[0].set_yticks(np.arange(-2.,2.5,1))
    #axs[0].set_ylim(-2.5,2.5)
    #axs[1].set_yticks(np.arange(-0.02,0.03,0.01))
    #axs[1].set_ylim(-.025,0.025)
    #axs[2].set_yticks(np.arange(-10,15,5))
    #axs[2].set_ylim(-15,15)

    # plt.suptitle('gm#{}'.format(ind))
    # plt.savefig(path+"comparison #"+str(ind+1)+".png", dpi=300, transparent=True, bbox_inches='tight')
    #plt.savefig("plot\comparison #"+str(ind+1)+".pdf", dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(path+"comparison #"+str(ind+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
for ii in np.argsort(np.max(np.abs(RES[ind_test,:,0]),axis=1))[[130,403,678]]:
    # print(np.max(np.abs(RES[ind_test[ii],:,0])),np.max(np.abs(predRES[ind_test[ii],:,0])))
    print(ind_test[ii])
    compare_plot(ind_test[ii],path="plot\\testing\\")


# In[20]:


for ii in ind_train[0:10]:
    compare_plot(ii,path="plot\\training\\")

for ii in ind_valid[0:10]:
    compare_plot(ii,path="plot\\validating\\")

# In[testining]

for ii in [75,1135,1920]:
    compare_plot(ii,path="plot\\testing\\")
    


# In[LSTM]
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

LSTM = tf.keras.Model(gm_inp, train_res)
LSTM.built = True
LSTM.load_weights(r'G:/research/11_surrogate_model2023_3/01_reference_LSTM/base_model_weights.h5')


with tf.device('/CPU:0'):
    predLSTM = LSTM(GM)
predLSTM  = np.array(predLSTM )
predLSTM  = destand(predLSTM )


# In[statistical compare]
# predRES.shape = (N,npoint,3)
# RES,shape = (N,npoint,3)
from metrics2 import cosSM, Cal_cdf, Plot_cdf, Cal_Residual, Cal_Peak, Cal_Amp, Cal_Eng, Cal_correlation, Plot_correlation, Cal_Dist
LOSS = []
    
mpl.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

pred_train, pred_test = predRES[ind_train,:,0], predRES[ind_test,:,0]
lstm_train, lstm_test = predLSTM[ind_train,:,0], predLSTM[ind_test,:,0]
true_train, true_test = RES[ind_train,:,0], RES[ind_test,:,0]


fig = plt.figure(figsize=(2*3,3*2.2))
gs = fig.add_gridspec(3,2)
axs = gs.subplots()
train_loss = Cal_Residual(pred_train,true_train)
test_loss = Cal_Residual(pred_test,true_test)
lstm_loss = Cal_Residual(lstm_test, true_test)
# Plot_cdf(train_loss,axs[0,0],'blue','$L_{residual}$','Train')
Plot_cdf(test_loss,axs[0,0],'red','$L_{residual}$','SPR-Net')
Plot_cdf(lstm_loss,axs[0,0],'blue','$L_{residual}$','LSTM',visuable=True)
axs[0,0].legend(fontsize=12,loc='lower right')
LOSS.append([test_loss,lstm_loss])

train_loss = Cal_Peak(pred_train,true_train)
test_loss = Cal_Peak(pred_test,true_test)
lstm_loss = Cal_Peak(lstm_test,true_test)
# Plot_cdf(train_loss,axs[0,1],'blue','$L_{peak}$','Train')
Plot_cdf(test_loss,axs[0,1],'red','$L_{peak}$','Test')
Plot_cdf(lstm_loss,axs[0,1],'blue','$L_{peak}$','Test',visuable=True)
LOSS.append([test_loss,lstm_loss])

train_loss = Cal_Amp(pred_train,true_train)
test_loss = Cal_Amp(pred_test,true_test)
lstm_loss = Cal_Amp(lstm_test,true_test)
# Plot_cdf(train_loss,axs[1,0],'blue','$\cal{A}$','Train')
Plot_cdf(test_loss,axs[1,0],'red','$\cal{A}$','Test')
Plot_cdf(lstm_loss,axs[1,0],'blue','$\cal{A}$','Test',visuable=True)
LOSS.append([test_loss,lstm_loss])

train_loss = Cal_Eng(pred_train,true_train)
test_loss = Cal_Eng(pred_test,true_test)
lstm_loss = Cal_Eng(lstm_test,true_test)
# Plot_cdf(train_loss,axs[1,1],'blue','$\cal{E}$','Train')
Plot_cdf(test_loss,axs[1,1],'red','$\cal{E}$','Test')
Plot_cdf(lstm_loss,axs[1,1],'blue','$\cal{E}$','Test',visuable=True)
LOSS.append([test_loss,lstm_loss])

train_r = Cal_correlation(pred_train,true_train)
test_r = Cal_correlation(pred_test,true_test)
lstm_r = Cal_correlation(lstm_test,true_test)
Plot_correlation([test_r,lstm_r],axs[2,0],color=['red','blue'],bound=0.4)
LOSS.append([test_r,lstm_r])

# train_mu,train_std = Cal_Dist(pred_train,true_train,bound=0.05,color='blue',axs=axs[2,1])

lstm_mu,lstm_std = Cal_Dist(lstm_test,true_test,bound=0.10,color='blue',axs=axs[2,1])
test_mu,test_std = Cal_Dist(pred_test,true_test,bound=0.10,color='red',axs=axs[2,1],leg=True,visuable=True)
LOSS.append([test_mu,test_std,lstm_mu,lstm_std])



plt.tight_layout()
plt.savefig("plot\loss0.svg", dpi=100, transparent=True, bbox_inches='tight')
plt.show()

open_file = open('plot\loss0.pkl', "wb")
pickle.dump(LOSS,open_file) # response record, neted list dt=2e-3
open_file.close()



# In[shapley]
np.random.seed(1)
tf.random.set_seed(1)
import tensorflow_probability as tfp
# Define the inputs and outputs using the Functional API for saving the model correctly
input_structure = tf.keras.Input(shape=(15,))
input_GM = tf.keras.Input(shape=(npoint, 2))
temp = model([input_structure, input_GM])
peak = tf.reduce_max(tf.abs(temp),axis=[1])

energy = tf.reduce_sum(tf.abs(temp),axis=[1])
energy1 = tfp.math.trapz(temp[:,:,1],-temp[:,:,0])[:,None]
energy2 = tfp.math.trapz(temp[:,:,3],-temp[:,:,2])[:,None]


residual = tf.reduce_mean(temp[:,:-10,:],axis=[1])
outputs = tf.concat([peak,energy,energy1,energy2,residual],axis=1)

# Create the functional model
model_explain = tf.keras.Model(inputs=[input_structure, input_GM], outputs=outputs)



# Check if the model has outputs
if model_explain.outputs is not None:
    print("Model outputs:", model_explain.outputs)
else:
    print("Model has no outputs. Check the model definition.")
    
    

import shap
x1,x2 = Structure[ind_valid,:],GM[ind_valid,:,:]
explainer = shap.GradientExplainer(model_explain, [x1,x2])
# Explain the model's predictions
shap_values = explainer.shap_values([x1,x2],nsamples=200)

with open('plot\\shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)
    

# In[]

import shap
x1,x2 = Structure[ind_valid,:],GM[ind_valid,:,:]
with open('plot\\shap_values.pkl', 'rb') as f:
    shap_values = pickle.load(f)
    
# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap_ranges = np.max(shap_values[0][0], axis=0) - np.min(shap_values[0][0], axis=0)
# sorted_idx = np.argsort(shap_ranges)[::-1]
# sorted_feature_names = np.array(feature_names)[sorted_idx]
# sorted_shap_values = shap_values[0][0][:, sorted_idx]
# sorted_x1 = x1[:, sorted_idx]
# shap.summary_plot(
#     sorted_shap_values, 
#     sorted_x1, 
#     feature_names=sorted_feature_names, 
#     show=False, 
#     cmap='seismic', 
#     plot_size=[6*cm,6*cm],
#     alpha=1,
#     sort=False,
#     max_display=10
# )
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
# ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
# colorbar.tick_params(labelsize=8)
# plt.xlim([-1.5,1.5])
# plt.xticks([-1.5,-0.75,0,0.75,1.5],fontsize=8)
# plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_columnPeak.svg',dpi=300)
# plt.show()


# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap.summary_plot(shap_values[1][0], x1,feature_names=feature_names,show=False,cmap=cmc.batlow, plot_size=[3.5,3.5],alpha=0.5)
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
# ax.set_xlabel('SHAP value', labelpad=2, fontsize=10)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=10, labelpad=-20)
# plt.xlim([-3,3])
# plt.xticks(np.arange(-3,4))
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_columnFPeak.svg',dpi=100)
# plt.show()

# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap_ranges = np.max(shap_values[2][0], axis=0) - np.min(shap_values[2][0], axis=0)
# sorted_idx = np.argsort(shap_ranges)[::-1]
# sorted_feature_names = np.array(feature_names)[sorted_idx]
# sorted_shap_values = shap_values[2][0][:, sorted_idx]
# sorted_x1 = x1[:, sorted_idx]
# shap.summary_plot(
#     sorted_shap_values, 
#     sorted_x1, 
#     feature_names=sorted_feature_names, 
#     show=False, 
#     cmap='seismic', 
#     plot_size=[6*cm,6*cm],
#     alpha=1,
#     sort=False,
#     max_display=10
# )
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
# ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
# colorbar.tick_params(labelsize=8)
# plt.xlim([-1.5,1.5])
# plt.xticks([-1.5,-0.75,0,0.75,1.5],fontsize=8)
# plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_bearingPeak.svg',dpi=300)
# plt.show()

# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap.summary_plot(shap_values[3][0], x1,feature_names=feature_names,show=False,cmap=cmc.batlow, plot_size=[3.5,3.5],alpha=0.5)
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
# ax.set_xlabel('SHAP value', labelpad=2, fontsize=10)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=10, labelpad=-20)
# plt.xlim([-2,2])
# plt.xticks(np.arange(-2,3,1))
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_bearingFPeak.svg',dpi=100)
# plt.show()


# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap_ranges = np.max(shap_values[4][0], axis=0) - np.min(shap_values[4][0], axis=0)
# sorted_idx = np.argsort(shap_ranges)[::-1]
# sorted_feature_names = np.array(feature_names)[sorted_idx]
# sorted_shap_values = shap_values[4][0][:, sorted_idx]
# sorted_x1 = x1[:, sorted_idx]
# shap.summary_plot(
#     sorted_shap_values, 
#     sorted_x1, 
#     feature_names=sorted_feature_names, 
#     show=False, 
#     cmap='seismic', 
#     plot_size=[6*cm,6*cm],
#     alpha=1,
#     sort=False,
#     max_display=10
# )
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
# ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
# colorbar.tick_params(labelsize=8)
# plt.xlim([-250,250])
# plt.xticks([-250,-125,0,125,250],fontsize=8)
# plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_columnEng.svg',dpi=300)
# plt.show()

# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap.summary_plot(shap_values[6][0], x1,feature_names=feature_names,show=False,cmap=cmc.batlow, plot_size=[3.5,3.5],alpha=0.5)
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
# ax.set_xlabel('SHAP value', labelpad=2, fontsize=10)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=10, labelpad=-20)
# # plt.xlim([-1.5,1.5])
# # plt.xticks(np.arange(-1.5,2,0.5))
# plt.tight_layout()
# # plt.savefig('plot\\sensitivity_columnFEng.svg',dpi=100)
# plt.show()

# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap_ranges = np.max(shap_values[6][0], axis=0) - np.min(shap_values[6][0], axis=0)
# # shap_ranges = np.std(shap_values[6][0], axis=0)
# sorted_idx = np.argsort(shap_ranges)[::-1]
# sorted_feature_names = np.array(feature_names)[sorted_idx]
# sorted_shap_values = shap_values[6][0][:, sorted_idx]
# sorted_x1 = x1[:, sorted_idx]
# shap.summary_plot(
#     sorted_shap_values, 
#     sorted_x1, 
#     feature_names=sorted_feature_names, 
#     show=False, 
#     cmap='seismic', 
#     plot_size=[6*cm,6*cm],
#     alpha=1,
#     sort=False,
#     max_display=10
# )
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
# ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
# colorbar.tick_params(labelsize=8)
# plt.xlim([-500,500])
# plt.xticks([-500,-250,0,250,500],fontsize=8)
# plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_bearingEng.svg',dpi=300)
# plt.show()


# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap.summary_plot(shap_values[7][0], x1,feature_names=feature_names,show=False,cmap=cmc.batlow, plot_size=[3.5,3.5],alpha=0.5)
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
# ax.set_xlabel('SHAP value', labelpad=2, fontsize=10)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=10, labelpad=-20)
# # plt.xlim([-1.5,1.5])
# # plt.xticks(np.arange(-1.5,2,0.5))
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_bearingFEng.svg',dpi=100)
# plt.show()

# plt.figure()
# feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
# shap_ranges = np.max(shap_values[8][0], axis=0) - np.min(shap_values[8][0], axis=0)
# sorted_idx = np.argsort(shap_ranges)[::-1]
# sorted_feature_names = np.array(feature_names)[sorted_idx]
# sorted_shap_values = shap_values[8][0][:, sorted_idx]
# sorted_x1 = x1[:, sorted_idx]
# shap.summary_plot(
#     sorted_shap_values, 
#     sorted_x1, 
#     feature_names=sorted_feature_names, 
#     show=False, 
#     cmap='seismic', 
#     plot_size=[6*cm,6*cm],
#     alpha=1,
#     sort=False,
#     max_display=10
# )
# ax = plt.gca()  # Get current axes
# for collection in ax.collections:
#     collection.set_sizes([3])
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
# ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.)
# colorbar = plt.gcf().axes[-1]  
# colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
# colorbar.tick_params(labelsize=8)
# plt.xlim([-200,200])
# plt.xticks([-200,-100,0,100,200],fontsize=8)
# plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
# plt.tight_layout()
# plt.savefig('plot\\sensitivity_columnEng - Hystersis.svg',dpi=300)
# plt.show()

plt.figure()
feature_names = [r"$L$",r"$H_c$",r"$W_d$",r"$\lambda$",r"$\rho_s$",r"$H_b$",r"$K_p$",r"$K_{rot}$",r"$K_t$",r"$f_c$", r"$k_b$",r"$\mu_b$",r"$\delta$",r"$m_s$",r"$\xi$"]
shap_ranges = np.max(shap_values[9][0], axis=0) - np.min(shap_values[9][0], axis=0)
sorted_idx = np.argsort(shap_ranges)[::-1]
sorted_feature_names = np.array(feature_names)[sorted_idx]
sorted_shap_values = shap_values[9][0][:, sorted_idx]
sorted_x1 = x1[:, sorted_idx]
shap.summary_plot(
    sorted_shap_values, 
    sorted_x1, 
    feature_names=sorted_feature_names, 
    show=False, 
    cmap='seismic', 
    plot_size=[6*cm,6*cm],
    alpha=1,
    sort=False,
    max_display=10
)
ax = plt.gca()  # Get current axes
for collection in ax.collections:
    collection.set_sizes([3])
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_xlabel('Normalized SHAP value', labelpad=2, fontsize=8)
for spine in ax.spines.values():
    spine.set_linewidth(1.)
colorbar = plt.gcf().axes[-1]  
colorbar.set_ylabel('Feature Value', fontsize=8, labelpad=-15)
colorbar.tick_params(labelsize=8)
plt.xlim([-50,50])
plt.xticks([-50,-25,0,25,50],fontsize=8)
plt.gca().set_xticklabels([-1.0,-0.5,0,0.5,1.0])
plt.tight_layout()
plt.savefig('plot\\sensitivity_bearingEng - Hystersis.svg',dpi=300)
plt.show()

# In[]
feature_names = [r"$L$ (m)",r"$H_c$ (m)",r"$W_d$ (m)",r"$\lambda$",r"$\rho_s$ (%)",r"$H_b$ (m)",r"$K_p (N/mm)$",r"$K_{rot}$ (N-m/rad)",r"$K_t$ (kN/mm)",r"$f_c$ (Mpa)", r"$k_b$(N/m/mm)",r"$\mu_b$",r"$\delta$ (mm)",r"$m_s$",r"$\xi$"]
for ii in range(len(feature_names)):
    plt.figure(figsize=(2.8,2))
    plt.hist(StructureInfo[:,ii],bins=8,rwidth=0.6,edgecolor='k',color='orange')
    plt.xlabel(feature_names[ii],labelpad=5)
    plt.ylabel('Number of samples',labelpad=2)
    plt.tick_params(axis='both',
                    which='minor',
                    length=2)
    plt.tick_params(axis='both',
                    which='major',
                    length=4)
    plt.tight_layout()
    plt.savefig("plot\\data_dist\\fig_"+str(ii)+".svg",dpi=100)
    plt.show()

