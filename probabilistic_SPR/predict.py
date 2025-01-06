# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:57:48 2024

@author: cning
"""
# In[1]:
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

import tensorflow.keras
print(tf.config.list_physical_devices())

runfile('load_data.py')

# In[15]:

model = tf.keras.models.load_model('final_model.h5')

with tf.device('/CPU:0'):
    mu,sig = model([Structure,GM])

predMean = destand(mu)
angle = ((predMean>0)-0.5)*2
mu = tf.abs(mu)+1e-9
sig = tf.abs(sig)+1e-9
mu_x = tf.math.log(mu**2/tf.math.sqrt(mu**2+sig**2))
sig_x = tf.sqrt(tf.math.log(1+sig**2/mu**2))

predRES = destand((tf.exp(mu_x))*angle) #median prediction
predUp = destand((tf.exp(mu_x+1*sig_x))*angle)
predLow =destand((tf.exp(mu_x-1*sig_x))*angle)
RES = RESTH.astype('float32')


lstm_model = tf.keras.models.load_model(r'G:/research/11_surrogate_model2023_3/01_reference_LSTM/best_LSTM_model.h5')
with tf.device('/CPU:0'):
    pred_lstm = lstm_model(GM)
pred_lstm = np.array(pred_lstm)
pred_lstm = destand(pred_lstm)

# In[122]:
def compare_plot(ind, path,dt = dt):
    
    fig = plt.figure(figsize=(6*cm,9*cm))
    gs = fig.add_gridspec(nrows=4,ncols=1,hspace=0.2,wspace=0.0,height_ratios=[1,1,1,1])
    axs = gs.subplots(sharex=True)
    for axis in ['top','bottom','left','right']:
        [ax.spines[axis].set_linewidth(0.5) for ax in axs]
    
    tmax = 40
    l = int(tmax/dt)
    x0 = np.arange(0,l)*dt
    scale = [100,0.004448,2.54,0.004448]
    ylabels = ["$\Delta_c(\%)$","$F_c (MN)$","$\gamma (cm)$","$F_b (MN)$"]
    
    for ii in np.arange(4):
        axs[ii].plot(x0,scale[ii]*RES[ind,:l,ii],linewidth=1,color='k',label='Ground truth',alpha=0.8)
        axs[ii].plot(x0,scale[ii]*predRES[ind,:l,ii],linewidth=0.5,linestyle='--',color='#DC0000FF',label='Predicted')
        axs[ii].plot(x0,scale[ii]*pred_lstm[ind,:l,ii],linewidth=0.5,linestyle='-.',color='#3C5488FF',label='LSTM')
        axs[ii].fill_between(x0,scale[ii]*predUp[ind,:l,ii],scale[ii]*predLow[ind,:l,ii],color='#F39B7FFF',alpha=1,edgecolor='None')
        axs[ii].margins(y=8*max(predUp[ind,50:l,0]))
        axs[ii].set_ylabel(ylabels[ii],labelpad=2)

    axs[3].set_xlabel('$t(s)$',labelpad=4)
    axs[3].set_xticks(np.arange(0,65,5))
    axs[3].set_xlim(0,20)

    axs[0].tick_params(labelbottom=False);axs[1].tick_params(labelbottom=False);axs[2].tick_params(labelbottom=False);
    fig.align_ylabels(axs[:])
    plt.savefig(path+str(ind+1)+".svg", dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    

for ii in np.argsort(np.max(np.abs(RES[ind_test,:,0]),axis=1))[[130,403,678]]:
    compare_plot(ind_test[ii],path="plot\\")

# In[statistical compare]

from metrics import Cal_cdf
mpl.rcParams['axes.linewidth'] = 0.5
xlabel = [r"$\Delta_c$ (%)", r"$F_c (MN)$",r"$\Delta_b (mm)$",r"$F_b (kN)$"]
scale  = [100, 0.004448, 25.4, 4.448]
xlim = [[0.05,3],[0.5,10],[5,150],[10,500]]
for ii in np.arange(4):
    true_test = RES[ind_test,:,ii]*scale[ii]
    pred_mean = predMean[ind_test,:,ii]*scale[ii]
    pred_median = predRES[ind_test,:,ii]*scale[ii]
    pred_top, pred_bot = predUp[ind_test,:,ii]*scale[ii],predLow[ind_test,:,ii]*scale[ii] #50 CI
    ref_test = pred_lstm[ind_test,:,ii]*scale[ii]
    
    ind = np.argmax(np.abs(pred_median),axis=1).tolist()
    cr = np.max(np.abs(true_test),axis=1)
    
    c00 = np.abs(pred_mean)[np.arange(len(ind)),ind]
    c0 = np.abs(pred_median)[np.arange(len(ind)),ind]
    c1 =pred_top[np.arange(len(ind)),ind]
    c2 = pred_bot[np.arange(len(ind)),ind]
    c3 = np.max(np.abs(ref_test),axis=1)
    up,low = [],[]
    for t1,t2 in zip(c1,c2):
        if abs(t1)>abs(t2):
            up.append(abs(t1))
            low.append(abs(t2))
        else:
            up.append(abs(t2))
            low.append(abs(t1))    
    
    plt.figure(figsize=(6*cm,3.7*cm))
    xr,yr = Cal_cdf(cr)
    print(np.percentile(xr,16),np.percentile(xr,50),np.percentile(xr,84))
    plt.plot(xr,yr,'k-',label='truth',lw=1.2)
    x,y = Cal_cdf(c3)
    print(np.percentile(x,16),np.percentile(x,50),np.percentile(x,84))
    plt.plot(x,y,color='#3C5488FF',linestyle='dashdot',label='lstm',lw=0.8)
    x1,y1 = Cal_cdf(up)
    print(np.percentile(x1,16),np.percentile(x1,50),np.percentile(x1,84))
    x2,y2 = Cal_cdf(low)
    print(np.percentile(x2,16),np.percentile(x2,50),np.percentile(x2,84))
    plt.fill_betweenx(y1, x1, x2,edgecolor=None,facecolor='gray',alpha=0.4,label='$\pm\sigma$')
    x0,y0 = Cal_cdf(c0)
    print(np.percentile(x0,16),np.percentile(x0,50),np.percentile(x0,84))
    plt.plot(x0,y0,color='#DC0000FF',linestyle='--',label='median',lw=0.8)
    
    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.ylabel('CDF',labelpad=2)
    plt.xlim(xlim[ii])
    
    plt.tight_layout()
    plt.savefig("plot\\"+xlabel[ii][1:-1]+".svg",dpi=300)
    plt.show()

# In[hysteresis energy disspation]
# scale  = [100, 0.004448, 25.4, 4.448]
StructureInfo = pd.read_excel(r'G:/research/11_surrogate_model2023_3/09_code/data/variables.xlsx').iloc[:,:16]
Hc = StructureInfo['H(m)'].values[:,None]

trueEC = np.trapz(RESTH[:,:,1]*4.448,-RESTH[:,:,0]*Hc) #kN*m
trueEB = np.trapz(RESTH[:,:,3]*4.448,-RESTH[:,:,2]*0.0254) #kN*m
truePC = np.max(np.abs(RESTH[:,:,0]),axis=1)
truePB = np.max(np.abs(RESTH[:,:,2]),axis=1)
refEC = np.trapz(pred_lstm[:,:,1]*4.448,-pred_lstm[:,:,0]*Hc) #kN*m
refEB = np.trapz(pred_lstm[:,:,3]*4.448,-pred_lstm[:,:,2]*0.0254) #kN*m
refPC = np.max(np.abs(pred_lstm[:,:,0]),axis=1)
refPB = np.max(np.abs(pred_lstm[:,:,2]),axis=1)


ec,eb = [],[]
for ii in range(1000):
    print(ii)
    eps = tf.random.normal([1950,1220,4],0,1,seed=2024)
    # eps = tf.tile(eps,[1,1,4])
    sam_i = destand((tf.exp(mu_x+eps*sig_x))*angle)
    ec.append(np.trapz(sam_i[:,::2,1]*4.448,-sam_i[:,::2,0]*Hc)) #shape 1950,1
    eb.append(np.trapz(sam_i[:,::2,3]*4.448,-sam_i[:,::2,2]*0.0254))

upEC,medEC,lowEC = np.percentile(ec,84,axis=0),np.percentile(ec,50,axis=0), np.percentile(ec,16,axis=0)
upEB,medEB,lowEB = np.percentile(eb,84,axis=0),np.percentile(eb,50,axis=0), np.percentile(eb,16,axis=0)

# medEC = np.trapz(predRES[:,:,1]*4.448,-predRES[:,:,0]*Hc)
# upEC = np.trapz(predUp[:,:,1]*4.448,-predUp[:,:,0]*Hc)
# lowEC = np.trapz(predLow[:,:,1]*4.448,-predLow[:,:,0]*Hc)
# medEB = np.trapz(predRES[:,:,3]*4.448,-predRES[:,:,2]*0.0254)
# upEB = np.trapz(predUp[:,:,3]*4.448,-predUp[:,:,2]*0.0254)
# lowEB = np.trapz(predLow[:,:,3]*4.448,-predLow[:,:,2]*0.0254)


# In[energy dissipation]
xlabel = [r"$E_c$ (kJ)", r"$E_b$ (kJ)"]
xlim = [[1,1e4],[1e-1,1e3]]
# xlim = [[0,4000],[0,150]]
xticks = [[1,10,1e2,1e3,1e4],[0.1,1,10,1e2,1e3]]


for ii in np.arange(len(xlabel)):
    if ii==0: 
        ind_temp=ind_test
    else:
        ind_temp = list(set(np.where(trueEB>0.1)[0]).intersection(set(ind_test)))
        # ind_temp = ind_test
        
    true_test = [trueEC[ind_temp],trueEB[ind_temp]][ii]
    ref_test = [refEC[ind_temp],refEB[ind_temp]][ii]
    pred_median = [medEC[ind_temp],medEB[ind_temp]][ii]
    pred_top = [upEC[ind_temp],upEB[ind_temp]][ii]
    pred_bot = [lowEC[ind_temp],lowEB[ind_temp]][ii] #50 CI
    
    plt.figure(figsize=(6*cm,3.7*cm))
    x,y = Cal_cdf(true_test)
    print(np.percentile(x,16),np.percentile(x,50),np.percentile(x,84))
    plt.plot(x,y,'k-',label='truth',lw=1.2)
    x,y = Cal_cdf(ref_test)
    print(np.percentile(x,16),np.percentile(x,50),np.percentile(x,84))
    plt.plot(x,y,linestyle='dashdot',color='#3C5488FF',label='lstm',lw=0.8)
    
    x1,y1 = Cal_cdf(pred_top)
    x2,y2 = Cal_cdf(pred_bot)
    print(np.percentile(x1,16),np.percentile(x1,50),np.percentile(x1,84))
    print(np.percentile(x2,16),np.percentile(x2,50),np.percentile(x2,84))
    plt.fill_betweenx(y1, x1, x2,edgecolor=None,facecolor='gray',alpha=0.4,label='$\pm\sigma$')
    
    x0,y0 = Cal_cdf(pred_median)
    print(np.percentile(x0,16),np.percentile(x0,50),np.percentile(x0,84))
    plt.plot(x0,y0,color='#DC0000FF',linestyle='--',label='median',lw=0.8)

    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.ylabel('CDF',labelpad=2)
    plt.xlim(xlim[ii])
    plt.xticks(xticks[ii])
    plt.minorticks_off()
    # plt.legend()
    plt.tight_layout()
    plt.savefig("plot\\sampling"+xlabel[ii][1:4]+".svg",dpi=300)
    plt.show()

# In[]
# 75,1135,1920

for ii in [75,1769,1350]:
    force,disp = RESTH[ii,:,1]*4.448,-RESTH[ii,:,0]*Hc[ii]
    predF,predD = predRES[ii,:,1]*4.448, -predRES[ii,:,0]*Hc[ii]
    predlowF,predlowD = predLow[ii,:,1]*4.448, -predLow[ii,:,0]*Hc[ii]
    predupF,predupD = predUp[ii,:,1]*4.448, -predUp[ii,:,0]*Hc[ii]
    lstmF,lstmD = pred_lstm[ii,:,1]*4.448, -pred_lstm[ii,:,0]*Hc[ii]
    realE = [np.trapz(force[0:jj],disp[0:jj]) for jj in np.arange(len(force))]
    predE = [np.trapz(predF[0:jj],predD[0:jj]) for jj in np.arange(len(predF))]
    predlowE = [np.trapz(predlowF[0:jj],predlowD[0:jj]) for jj in np.arange(len(predlowF))]
    predupE = [np.trapz(predupF[0:jj],predupD[0:jj]) for jj in np.arange(len(predupF))]
    lstmE  = [np.trapz(lstmF[0:jj],lstmD[0:jj]) for jj in np.arange(len(lstmF))]
    
    fig = plt.figure(figsize=(9*cm,3.5*cm))
    gs = fig.add_gridspec(nrows=1,ncols=2,wspace=0.3,width_ratios=[1,1])
    axs = gs.subplots()
    axs[0].plot(np.arange(npoint)*dt,realE,'k',lw=1.)
    axs[0].plot(np.arange(npoint)*dt,predE,linestyle='-',color='#DC0000FF',lw=0.5)
    axs[0].plot(np.arange(npoint)*dt,lstmE,linestyle='dashdot',color='#3C5488FF',label='lstm',lw=0.5)
    axs[0].fill_between(np.arange(npoint)*dt,predlowE,predupE,color='#F39B7FFF',alpha=0.5,edgecolor='None')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('t (s)',fontsize=8,labelpad=2)
    axs[0].set_ylabel(r'$E_c$ (kJ)',fontsize=8,labelpad=2)
    axs[0].set_ylim(1e-2,)
    axs[0].set_xlim([0,40])
    axs[0].set_xticks(np.arange(0,50,10))
    axs[0].minorticks_off()
    
    axs[1].plot(-RESTH[ii,:,0]*100,RESTH[ii,:,1]*0.004448,'k',lw=0.8,label='truth',alpha=0.8)
    axs[1].plot(-predRES[ii,:,0]*100,predRES[ii,:,1]*0.004448,color='#DC0000FF',linestyle='-',lw=0.5,label='meidin')
    axs[1].plot(-pred_lstm[ii,:,0]*100,pred_lstm[ii,:,1]*0.004448,linestyle='--',color='#3C5488FF',label='lstm',lw=0.5)
    axs[1].set_xlabel('$\Delta_c $ (%)',fontsize=8,labelpad=2)
    axs[1].set_ylabel('$F_c$ (MN)',fontsize=8,labelpad=2)
    plt.tight_layout()
    plt.savefig("plot\\Hystersis_"+str(ii+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
    compare_plot(ii,path="plot\\resp_")

# In[PSDM]
import scipy
from sklearn.linear_model import LinearRegression
Sa = pd.read_excel(r"G:/research/11_surrogate_model2023_3/09_code/data/variables.xlsx")['PGA'].values

Dy = np.loadtxt(r'G:\research\08_Active_learning_2\02_extract_data\\Dy_75.dat')[:,1]*25.4/1000
Hc = pd.read_excel(r'G:\research\11_surrogate_model2023_3\00_data\variables.xlsx')['H(m)'].values
Dy = Dy/Hc # yielding drift

def LSQ(IM,EDP,limit,std=0.25):
    log_IM = np.log(IM).reshape([-1,1])
    log_EDP = np.log(EDP).reshape([-1,1])
    
    xGrid = np.linspace(0.01, 3, 100)
    LR = LinearRegression()
    # LR = make_pipeline(StandardScaler(), LinearRegression())
    LR.fit(log_IM,log_EDP) #only Sa
    
    designMatrix=np.log(xGrid).reshape([-1,1])
    yPred = LR.predict(log_IM)
    trueStd = np.sqrt(sum((log_EDP-yPred)**2/IM.shape[0]))
    trueY = LR.predict(designMatrix)
    y = scipy.stats.norm.cdf((trueY-np.log(limit))/np.sqrt(trueStd**2+std**2))
    return xGrid,y.reshape(-1),trueY.reshape(-1),trueStd

EDP_threshold = [1,2,3,4]
EDP_std = [0.25,0.25,0.47,0.47]
nState = len(EDP_threshold)
ind = list(np.arange(N))

# fig,ax = plt.subplots(figsize=(6*cm,3.7*cm))
fig,ax = plt.subplots(figsize=(6*cm,5*cm))
markers=['o','x','^','s']
alpha = [0.6,0.5,0.4,0.3]

for ss in range(nState):
    xGrid,true_frag,_,true_beta = LSQ(Sa[ind], truePC[ind]/Dy[ind], EDP_threshold[ss],EDP_std[ss])
    _,ref_frag,_,ref_beta = LSQ(Sa[ind], refPC[ind]/Dy[ind], EDP_threshold[ss],EDP_std[ss])
    i0 = np.argmax(np.abs(predRES[:,:,0]),axis=1).tolist()
    _,med_frag,_,med_beta = LSQ(Sa[ind], np.abs(predRES[ind,:,0])[np.arange(len(i0)),i0]/Dy[ind], EDP_threshold[ss],EDP_std[ss])
    _,mean_frag,_,mean_beta = LSQ(Sa[ind], np.abs(predMean[ind,:,0])[np.arange(len(i0)),i0]/Dy[ind], EDP_threshold[ss],EDP_std[ss])
    _,up_frag,_,up_beta = LSQ(Sa[ind], np.abs(predUp[ind,:,0])[np.arange(len(i0)),i0]/Dy[ind], EDP_threshold[ss],EDP_std[ss])
    _,low_frag,_,low_beta = LSQ(Sa[ind], np.abs(predLow[ind,:,0])[np.arange(len(i0)),i0]/Dy[ind], EDP_threshold[ss],EDP_std[ss])

    p1, = ax.plot(xGrid,true_frag,'k',lw=1.,marker=markers[ss],markersize=2,markevery=10)
    p2, = ax.plot(xGrid,med_frag,color='#DC0000FF',linestyle='--',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    p3, = ax.plot(xGrid,ref_frag,color='#3C5488FF',linestyle='dashdot',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    # p3, = ax.plot(xGrid,mean_frag,'b--',lw=0.8)
    p4 = ax.fill_between(xGrid,np.min([low_frag,up_frag],axis=0),np.max([low_frag,up_frag],axis=0),color='gray',alpha=alpha[ss])
    
# ax.legend((p1,p2,p4),('ground truth','median','$\pm \sigma$'),loc='lower right',fontsize=10)
ax.set_xlabel('PGA [g]',labelpad=2)
ax.set_ylabel('P(D>C|IM)',labelpad=2)
# ax.grid(which='major',axis='y',linewidth=0.5)
# ax.grid(which='major',axis='x',linewidth=0.3)
plt.tight_layout()
plt.savefig('plot\\column_frag.svg')
plt.show()


# In[bearing disp. fragility]
EDP_threshold = np.array([25.4,101.6])*0.0393701 #mm to in
nState = len(EDP_threshold)

fig,ax = plt.subplots(figsize=(6*cm,5*cm))
markers=['o','x','^','s']
for ss in range(nState):
    xGrid,true_frag,_,true_beta = LSQ(Sa, truePB, EDP_threshold[ss])
    _,ref_frag,_,ref_beta = LSQ(Sa, refPB, EDP_threshold[ss])
    i0 = np.argmax(np.abs(predRES[:,:,2]),axis=1).tolist()
    _,med_frag,_,med_beta = LSQ(Sa, np.abs(predRES[:,:,2])[np.arange(len(i0)),i0], EDP_threshold[ss])
    _,mean_frag,_,med_beta = LSQ(Sa, np.abs(predMean[:,:,2])[np.arange(len(i0)),i0], EDP_threshold[ss])
    _,up_frag,_,up_beta = LSQ(Sa, np.abs(predUp[:,:,2])[np.arange(len(i0)),i0], EDP_threshold[ss])
    _,low_frag,_,low_beta = LSQ(Sa, np.abs(predLow[:,:,2])[np.arange(len(i0)),i0], EDP_threshold[ss])

    p1, = ax.plot(xGrid,true_frag,'k',lw=1.,marker=markers[ss],markersize=2,markevery=10)
    p2, = ax.plot(xGrid,med_frag,color='#DC0000FF',linestyle='--',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    p3, = ax.plot(xGrid,ref_frag,color='#3C5488FF',linestyle='dashdot',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    # p3, = ax.plot(xGrid,mean_frag,'b--',lw=0.8)
    p4 = ax.fill_between(xGrid,np.min([low_frag,up_frag],axis=0),np.max([low_frag,up_frag],axis=0),color='gray',alpha=alpha[ss])
# ax.legend((p1,p2,p3,p4),('ground truth','median','mean','$\pm \sigma$'),loc='lower right',fontsize=10)
ax.set_xlabel('PGA [g]',labelpad=2)
ax.set_ylabel('P(D>C|IM)',labelpad=2)
plt.tight_layout()
plt.savefig('plot\\bearing_frag.svg')
plt.show()

# In[column Energy]
Fy = np.loadtxt(r'G:\research\08_Active_learning_2\02_extract_data\\Strength_75.dat')[:,1]*4.448*Hc

# EDP_threshold = np.array([0.65,1.9,3.4,5.2])
EDP_threshold = np.array([1,1,1,1])
Dy_threshold = [1,2,3,4]
nState = len(EDP_threshold)

# trueEC is in kip*drift
# truePC is in drift
# Fy is in kip
# Dy is in drift
# Hc is in meter
beta = 0.05
fig,ax = plt.subplots(figsize=(6*cm,5*cm))
for ss in range(nState):
    # for ss in range(nState):
    i0 = np.argmax(np.abs(predRES[:,:,0]),axis=1).tolist()
    edp = truePC/(Dy*Dy_threshold[ss])+beta*trueEC/(Fy*Dy*Dy_threshold[ss])
    xGrid,true_frag,_,true_beta = LSQ(Sa, edp, EDP_threshold[ss])
    edp = refPC/(Dy*Dy_threshold[ss])+beta*refEC/(Fy*Dy*Dy_threshold[ss])
    _,ref_frag,_,ref_beta = LSQ(Sa, edp, EDP_threshold[ss])
    edp = np.abs(predRES[:,:,0])[np.arange(len(i0)),i0]/(Dy*Dy_threshold[ss])+beta*medEC/(Fy*Dy*Dy_threshold[ss])
    _,med_frag,_,med_beta = LSQ(Sa,edp , EDP_threshold[ss])
    edp = np.abs(predMean[:,:,0])[np.arange(len(i0)),i0]/(Dy*Dy_threshold[ss])+beta*medEC/(Fy*Dy*Dy_threshold[ss])
    _,mean_frag,_,med_beta = LSQ(Sa,edp , EDP_threshold[ss])
    edp = np.abs(predUp[:,:,0])[np.arange(len(i0)),i0]/(Dy*Dy_threshold[ss])+beta*upEC/(Fy*Dy*Dy_threshold[ss])
    _,up_frag,_,up_beta = LSQ(Sa,edp , EDP_threshold[ss])
    edp = np.abs(predLow[:,:,0])[np.arange(len(i0)),i0]/(Dy*Dy_threshold[ss])+beta*lowEC/(Fy*Dy*Dy_threshold[ss])
    _,low_frag,_,low_beta = LSQ(Sa,edp , EDP_threshold[ss])

    p1, = ax.plot(xGrid,true_frag,'k',lw=1.,marker=markers[ss],markersize=2,markevery=10)
    p2, = ax.plot(xGrid,med_frag,color='#DC0000FF',linestyle='--',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    p3, = ax.plot(xGrid,ref_frag,color='#3C5488FF',linestyle='dashdot',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
    # p3, = ax.plot(xGrid,mean_frag,'b--',lw=0.8)
    p4 = ax.fill_between(xGrid,np.min([low_frag,up_frag],axis=0),np.max([low_frag,up_frag],axis=0),color='gray',alpha=alpha[ss])
# ax.legend((p1,p2,p3,p4),('ground truth','median','mean','$\pm \sigma$'),loc='lower right',fontsize=10)
ax.set_xlabel('PGA [g]',labelpad=2)
ax.set_ylabel('P(D>C|IM)',labelpad=2)
plt.tight_layout()
plt.savefig('plot\\parker_frag.svg')
plt.show()
