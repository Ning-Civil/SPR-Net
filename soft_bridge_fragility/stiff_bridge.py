# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:57:48 2024

@author: cning
"""
# In[import library]:
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import pickle


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

cm = 1/2.54
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

import tensorflow.keras
print(tf.config.list_physical_devices())

# In[load dataset]:
runfile('load_data.py')

# for soft structure
# L = 120ft = 36.576m; lambda = 25ft/48in = 6.25 , abutment_gap = 1.2in = 30.48mm #new bridge
# for median structure
# L = 100ft = 30.48m; lambda = 20ft/48in = 5 , abutment_gap = 0.9in = 22.86mm #new bridge
# L = 90.99ft = 27.734m; lambda = 22.382ft/48in = 5.596 , abutment_gap = 1.969in = 50.0126mm #AL2
# for stiff structure
# L = 80ft = 24.384m; lambda = 20ft/60in = 4 , abutment_gap = 0.6in = 15.24mm #new bridge
# for stiff structure
# L = 80ft = 24.384m; lambda = 20ft/60in = 4 , abutment_gap = 0.6in = 15.24mm #new bridge
# L = 96.138ft = 29.303m; lambda = 18.456ft/60in = 3.6912 , abutment_gap = 0.872in = 22.1488mm #AL2


# 80% shear force
# Dy = 1.2547817*25.4/1000 # AL2
# Hc = 18.456*12*25.4/1000 # AL2
# Dy = Dy/Hc*100 # yielding drift percentage
# Fy = 6.6117100e+02*4.448/1000 #AL2

# 75% shear force
Dy = 1.1995557e+00*25.4/1000 # AL2
Hc = 18.456*12*25.4/1000 # AL2
Dy = Dy/Hc*100 # yielding drift percentage
Fy = 6.6117100e+02*4.448/1000 #AL2

FeatureName = ["L(m)","lambda","abut_gap(mm)"]
StructureInfo = np.zeros((N,3))
StructureInfo[:,0] = 29.303
StructureInfo[:,1] = 3.6912
StructureInfo[:,2] = 22.15
Structure = (StructureInfo-meanStructure)/stdStructure
print(StructureInfo.shape,FeatureName)

RESTH = stiff_RESTH

# In[predicting]:

model = tf.keras.models.load_model('final_model.h5')

with tf.device('/CPU:0'):
    mu,sig = model([Structure,GM])

predMean = destand(mu)
angle = ((predMean>0)-0.5)*2
mu = tf.abs(mu)+1e-9
sig = tf.abs(sig)+1e-9
mu_x = tf.math.log(mu**2/tf.math.sqrt(mu**2+sig**2))
sig_x = tf.sqrt(tf.math.log(1+sig**2/mu**2))

predMedian = destand((tf.exp(mu_x))*angle) #median prediction
predUp = destand((tf.exp(mu_x+1*sig_x))*angle)
predLow =destand((tf.exp(mu_x-1*sig_x))*angle)


lstm_model = tf.keras.models.load_model(r'G:/research/11_surrogate_model2023_3/01_reference_LSTM/best_LSTM_model.h5')
with tf.device('/CPU:0'):
    pred_lstm = lstm_model(GM)
pred_lstm = np.array(pred_lstm)
pred_lstm = destand(pred_lstm)

# In[TS plot]:

def compare_plot(ind, path,dt = dt):
    
    fig = plt.figure(figsize=(6*cm,9*cm))
    gs = fig.add_gridspec(nrows=4,ncols=1,hspace=0.2,wspace=0.0,height_ratios=[1,1,1,1])
    axs = gs.subplots(sharex=True)
    for axis in ['top','bottom','left','right']:
        [ax.spines[axis].set_linewidth(0.5) for ax in axs]
    
    tmax = 40
    l = int(tmax/dt)
    x0 = np.arange(0,l)*dt
    scale = [1,1,1,1]
    ylabels = ["$\Delta_c(\%)$","$F_c (MN)$","$\gamma (cm)$","$F_b (MN)$"]
    
    for ii in np.arange(4):
        axs[ii].plot(x0,scale[ii]*RESTH[ind,:l,ii],linewidth=1,color='k',label='Ground truth',alpha=0.8)
        axs[ii].plot(x0,scale[ii]*predMedian[ind,:l,ii],linewidth=0.5,linestyle='--',color='#DC0000FF',label='Predicted')
        axs[ii].plot(x0,scale[ii]*pred_lstm[ind,:l,ii],linewidth=0.5,linestyle='-.',color='#3C5488FF',label='LSTM')
        axs[ii].fill_between(x0,scale[ii]*predUp[ind,:l,ii],scale[ii]*predLow[ind,:l,ii],color='#F39B7FFF',alpha=1,edgecolor='None')
        axs[ii].margins(y=0.08*max(predUp[ind,50:l,0]))
        axs[ii].set_ylabel(ylabels[ii],labelpad=2)

    axs[3].set_xlabel('$t(s)$',labelpad=4)
    axs[3].set_xticks(np.arange(0,65,5))
    axs[3].set_xlim(0,20)

    axs[0].tick_params(labelbottom=False);axs[1].tick_params(labelbottom=False);axs[2].tick_params(labelbottom=False);
    fig.align_ylabels(axs[:])
    plt.savefig(path+str(ind+1)+".svg", dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


for ii in np.random.choice(N,5,False):
    compare_plot(ii,path="02_stiff_plot\\")

# In[peak compare]
from metrics import Cal_cdf

xlabel = [r"$\Delta_c$ (%)", r"$F_c (MN)$",r"$\Delta_b (cm)$",r"$F_b (MN)$"]
xlim = [[0.05,3],[0.5,10],[0.5,15],[0.010,0.500]]
# scale = [1,1,1,1]
for ii in np.arange(4):
    true_test = RESTH[:,:,ii]
    pred_mean = predMean[:,:,ii]
    pred_median = predMedian[:,:,ii]
    pred_top, pred_bot = predUp[:,:,ii],predLow[:,:,ii] #50 CI
    
    ind = np.argmax(np.abs(pred_median),axis=1).tolist()
    cr = np.max(np.abs(true_test),axis=1)
    
    c00 = np.abs(pred_mean)[np.arange(len(ind)),ind]
    c0 = np.abs(pred_median)[np.arange(len(ind)),ind]
    c1 =pred_top[np.arange(len(ind)),ind]
    c2 = pred_bot[np.arange(len(ind)),ind]
    up,low = [],[]
    for t1,t2 in zip(c1,c2):
        if abs(t1)>abs(t2):
            up.append(abs(t1))
            low.append((t1*t2>=0)*abs(t2))
        else:
            up.append(abs(t2))
            low.append((t1*t2>=0)*abs(t1))    
    
    plt.figure(figsize=(2.5,2.2))
    x,y = Cal_cdf(cr)
    plt.plot(x,y,'k-',label='truth',lw=1.2)
    x1,y1 = Cal_cdf(up)
    x2,y2 = Cal_cdf(low)
    plt.fill_betweenx(y1, x1, x2,edgecolor=None,facecolor='gray',alpha=0.5,label='$\pm\sigma$')
    x0,y0 = Cal_cdf(c0)
    plt.plot(x0,y0,'--r',label='median',lw=0.8)
    # x00,y00 = Cal_cdf(c00)
    # plt.plot(x00,y00,'--b',label='mean')
    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylabel('CDF',labelpad=2)
    plt.grid(which='major',axis='y',linewidth=0.5)
    plt.grid(which='both',axis='x',linewidth=0.3)
    plt.xlim(xlim[ii])
    # plt.legend()
    plt.tight_layout()
    plt.savefig("02_stiff_plot\\"+xlabel[ii][1:-1]+".svg",dpi=100)
    plt.show()
    
# In[hysteresis]
for ii in [212,130,630]:
    force,disp = RESTH[ii,:,1],-RESTH[ii,:,0]*Hc
    predF,predD = predMedian[ii,:,1], -predMedian[ii,:,0]*Hc
    predlowF,predlowD = predLow[ii,:,1], -predLow[ii,:,0]*Hc
    predupF,predupD = predUp[ii,:,1], -predUp[ii,:,0]*Hc
    realE = [np.trapz(force[0:jj],disp[0:jj]) for jj in np.arange(len(force))]
    predE = [np.trapz(predF[0:jj],predD[0:jj]) for jj in np.arange(len(predF))]
    predlowE = [np.trapz(predlowF[0:jj],predlowD[0:jj]) for jj in np.arange(len(predlowF))]
    predupE = [np.trapz(predupF[0:jj],predupD[0:jj]) for jj in np.arange(len(predupF))]
    
    plt.figure(figsize=(2.2,1.8))
    plt.plot(np.arange(npoint)*dt,realE,'k',lw=1.2)
    plt.plot(np.arange(npoint)*dt,predE,'r-',lw=0.8)
    plt.fill_between(np.arange(npoint)*dt,predlowE,predupE,color='gray',alpha=0.8)
    plt.yscale('log')
    plt.xlabel('t(s)',labelpad=2)
    plt.ylabel(r'$MN\bullet m$',labelpad=2)
    # plt.ylim(1e-2,)
    plt.xlim([0,40])
    plt.savefig("02_stiff_plot\\Ec_"+str(ii+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(2.2,1.8))
    plt.plot(-RESTH[ii,:,0],RESTH[ii,:,1],'k',lw=1.2,label='truth')
    plt.plot(-predMedian[ii,:,0],predMedian[ii,:,1],'r',lw=0.8,label='meidin')
    # plt.plot(-predLow[ii,:,0]*100,predLow[ii,:,1],'gray',lw=0.8,alpha=0.5,label='$\sigma$')
    # plt.plot(-predUp[ii,:,0]*100,predUp[ii,:,1],'gray',lw=0.8,alpha=0.5,label='$\sigma$')
    plt.xlabel('$\Delta_c $ (%)')
    plt.ylabel('$F_c$ (MN)')
    plt.savefig("02_stiff_plot\\Hystersis_"+str(ii+1)+".svg", dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    
    
    # compare_plot(ii,path="02_stiff_plot\\resp_")

# In[energy compare]

trueEC = np.trapz(RESTH[:,:,1],-RESTH[:,:,0]*Hc) #MN*m
trueEB = np.trapz(RESTH[:,:,3],-RESTH[:,:,2]) #MN*cm


medEC = np.trapz(predMedian[:,:,1],-predMedian[:,:,0]*Hc)
upEC = np.trapz(predUp[:,:,1],-predUp[:,:,0]*Hc)
lowEC = np.trapz(predLow[:,:,1],-predLow[:,:,0]*Hc)
medEB = np.trapz(predMedian[:,:,3],-predMedian[:,:,2])
upEB = np.trapz(predUp[:,:,3],-predUp[:,:,2])
lowEB = np.trapz(predLow[:,:,3],-predLow[:,:,2])

xlabel = [r"$E_c (MN\bullet m)$", r"$E_b (MN\bullet cm)$"]
xlim = [[0.1,1e3],[0.1,1e1]]
xticks = [[1,10,1e2,1e3],[0.1,1,10,]]


for ii in np.arange(len(xlabel)):
    if ii==0: 
        ind_temp=np.arange(N)
    else:
        ind_temp = list(set(np.where(trueEB>0.1)[0]))
        
    true_test = [trueEC[ind_temp],trueEB[ind_temp]][ii]
    pred_median = [medEC[ind_temp],medEB[ind_temp]][ii]
    pred_top = [upEC[ind_temp],upEB[ind_temp]][ii]
    pred_bot = [lowEC[ind_temp],lowEB[ind_temp]][ii] #50 CI
    
    plt.figure(figsize=(2.5,2.2))
    x,y = Cal_cdf(true_test)
    plt.plot(x,y,'k-',label='truth',lw=1.2)
    
    x1,y1 = Cal_cdf(pred_top)
    x2,y2 = Cal_cdf(pred_bot)
    plt.fill_betweenx(y1, x1, x2,edgecolor=None,facecolor='gray',alpha=0.5,label='$\pm\sigma$')
    
    x0,y0 = Cal_cdf(pred_median)
    plt.plot(x0,y0,'--r',label='median',lw=0.8)

    plt.xscale('log')
    plt.xlabel(xlabel[ii],labelpad=2)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylabel('CDF',labelpad=2)
    plt.grid(which='major',axis='y',linewidth=0.5)
    plt.grid(which='major',axis='x',linewidth=0.3)
    plt.xlim(xlim[ii])
    plt.xticks(xticks[ii])
    plt.minorticks_off()
    # plt.legend()
    plt.tight_layout()
    # plt.savefig("02_stiff_plot\\"+xlabel[ii][1:4]+".svg",dpi=100)
    plt.show()
# In[sampling]

Pc,Pb,Ec = [],[],[]
for ii in range(1000):
    print(ii)
    eps = tf.random.normal([1950,1220,4],0,1,seed=2024)
    # eps = tf.tile(eps,[1,1,4])
    temp = destand((tf.exp(mu_x+eps*sig_x))*angle)
    ind = np.argmax(np.abs(predMedian[:,:,0]),axis=1).tolist()
    Pc.append(np.abs(temp[:,:,0])[np.arange(len(ind)),ind]) #drift
    ind = np.argmax(np.abs(predMedian[:,:,2]),axis=1).tolist()
    Pb.append(np.abs(temp[:,:,2])[np.arange(len(ind)),ind]) #inch
    Ec.append(np.trapz(temp[:,::2,1],-temp[:,::2,0])) #MN*drift
    
# In[]

import scipy
from sklearn.linear_model import LinearRegression

def LSQ(IM,EDP,limit,std):
    mask = EDP>0
    log_IM = np.log(IM[mask]).reshape([-1,1])
    log_EDP = np.log(EDP[mask]).reshape([-1,1])
    
    xGrid = np.linspace(0.01, 3, 100)
    LR = LinearRegression()
    LR.fit(log_IM,log_EDP) #only Sa
    
    designMatrix=np.log(xGrid).reshape([-1,1])
    yPred = LR.predict(log_IM)
    trueStd = np.sqrt(sum((log_EDP-yPred)**2/IM.shape[0]))
    trueY = LR.predict(designMatrix)
    frag = np.zeros((len(xGrid),len(limit)))
    for ii in range(len(limit)):
        frag[:,ii] = scipy.stats.norm.cdf((trueY-np.log(limit[ii]))/np.sqrt(trueStd**2+std[ii]**2))[:,0]
    return xGrid,frag,trueY.reshape(-1),trueStd


    
def fragility(Sa,true_EDP,sample_EDP,name,EDP_threshold,EDP_std,dr=None,ref_EDP=None,hazard=None):
    p1,p2 = 16,84
    nState = len(EDP_threshold)
    xGrid,true_frag,_,_ = LSQ(Sa,true_EDP, EDP_threshold,EDP_std)
    _,ref_frag,_,_ = LSQ(Sa,ref_EDP, EDP_threshold,EDP_std)
    alpha = [0.6,0.5,0.4,0.3]
    nx = len(xGrid)
    up_state,med_state,low_state = np.zeros((nx,nState)),np.zeros((nx,nState)),np.zeros((nx,nState))
    true_loss =  np.zeros((nx,nState))
    true_state = np.zeros((nx,nState))
    ref_loss =  np.zeros((nx,nState))
    ref_state = np.zeros((nx,nState))
    
    up_EDP = np.percentile(sample_EDP,p2,axis=0)
    med_EDP = np.percentile(sample_EDP,50,axis=0)
    low_EDP = np.percentile(sample_EDP,p1,axis=0)
    _,up_frag,_,_ = LSQ(Sa,up_EDP,EDP_threshold,EDP_std)
    _,med_frag,_,_ = LSQ(Sa,med_EDP,EDP_threshold,EDP_std)
    _,low_frag,_,_ = LSQ(Sa,low_EDP,EDP_threshold,EDP_std)

    for jj in range(nState-1):
        true_state[:,jj] = true_frag[:,jj]-true_frag[:,jj+1]
        ref_state[:,jj] = ref_frag[:,jj]-ref_frag[:,jj+1]
        up_state[:,jj] = up_frag[:,jj]-up_frag[:,jj+1]
        med_state[:,jj] = med_frag[:,jj]-med_frag[:,jj+1]
        low_state[:,jj] = low_frag[:,jj]-low_frag[:,jj+1]
    true_state[:,-1] = true_frag[:,-1]
    ref_state[:,-1] =ref_frag[:,-1]
    up_state[:,-1] = up_frag[:,-1]
    med_state[:,-1] = med_frag[:,-1]
    low_state[:,-1] = low_frag[:,-1]
    
    true_loss = true_state*dr
    ref_loss = ref_state*dr
    up_loss = up_state*dr
    med_loss = med_state*dr
    low_loss = low_state*dr
    
    true_risk= sum(hazard(xGrid)@true_loss[:,:])*100
    ref_risk = sum(hazard(xGrid)@ref_loss[:,:])*100
    up_risk=sum(hazard(xGrid)@up_loss[:,:])*100
    med_risk=sum(hazard(xGrid)@med_loss[:,:])*100
    low_risk=sum(hazard(xGrid)@low_loss[:,:])*100

    fig,ax = plt.subplots(figsize=(6*cm,5*cm))
    markers=['o','x','^','s']
    for ss in range(nState):
        l1, = ax.plot(xGrid,true_frag[:,ss],'k',lw=1.,marker=markers[ss],markersize=2,markevery=10)
        l2, = ax.plot(xGrid,med_frag[:,ss],color='#DC0000FF',linestyle='--',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
        l3, = ax.plot(xGrid,ref_frag[:,ss],color='#3C5488FF',linestyle='dashdot',lw=0.8,marker=markers[ss],markersize=2,markevery=10)
        l4 = ax.fill_between(xGrid,np.min([low_frag[:,ss],up_frag[:,ss]],axis=0),np.max([low_frag[:,ss],up_frag[:,ss]],axis=0),
                             edgecolor=None,color='gray',alpha=alpha[ss])
    
    # ax.legend((p1,p2,p4),('ground truth','median','$\pm \sigma$'),loc='lower right',fontsize=10)
    ax.set_xlabel('PGA [g]',labelpad=2)
    ax.set_ylabel('P(D>C|IM)',labelpad=2)
    plt.tight_layout()
    plt.savefig("02_stiff_plot\\"+name+"_frag75.svg")
    plt.show()
    
    fig,ax = plt.subplots(figsize=(6*cm,5*cm))
    for ss in range(nState):
        ax.plot(xGrid,true_state[:,ss],'k',lw=1.0,marker=markers[ss],markersize=1,markevery=10)
        ax.plot(xGrid,ref_state[:,ss],color='#3C5488FF',linestyle='dashdot',lw=0.8,marker=markers[ss],markersize=1,markevery=10)
        ax.plot(xGrid,med_state[:,ss],color='#DC0000FF',linestyle='--',lw=0.8,marker='o',markersize=1,markevery=10)
        ax.fill_between(xGrid,low_state[:,ss],up_state[:,ss],
                        edgecolor=None,color='gray',alpha=0.4)
    ax.set_ylim(0,)
    ax.set_xlabel('PGA(g)',labelpad=2)
    ax.set_ylabel('Damage Probability',labelpad=2)
    plt.tight_layout()
    plt.savefig("02_stiff_plot\\"+name+"_damage75.svg")
    plt.show()
    
    fig,ax = plt.subplots(figsize=(6*cm,5*cm))
    bars = ["Prediction","Ground truth","LSTM"]
    x_pos = [1,3,5]
    ax.bar(x_pos[0], med_risk, color='#a00000', yerr=[[med_risk-low_risk],[up_risk-med_risk]], capsize=5, 
           label=bars[0],width=1,edgecolor='k',lw=0.8,error_kw={'elinewidth':0.8,'capthick':1})
    ax.bar(x_pos[1],true_risk,color='#f2c45f',label=bars[1],width=1,edgecolor='k',lw=0.8)
    ax.bar(x_pos[2],ref_risk,color='#1a80bb',label=bars[2],width=1,edgecolor='k',lw=0.8)
    print(low_risk,med_risk,up_risk,true_risk,ref_risk)
    ax.set_ylabel('Normalized loss ratio (%)',labelpad=2)
    ax.set_ylim([0,round(np.max([low_risk,up_risk,true_risk,ref_risk])*1.5*100)/100])
    ax.tick_params(axis='x', which='major', pad=4)
    ax.set_xticks(x_pos,bars)
    plt.tight_layout()

    plt.savefig("02_stiff_plot\\"+name+"_risk75.svg")
    plt.show()




cost_column = 0.42
def hazard(pga):
    alpha,beta,gamma = 63.7280,42.4749,35.4039
    return alpha * np.exp(beta / (np.log(pga) - np.log(gamma)))

Sa = PGA
true_EDP = np.max(np.abs(RESTH[:,:,0]),axis=1)/Dy
ref_EDP = np.max(np.abs(pred_lstm[:,:,0]),axis=1)/Dy
sample_EDP = np.array(Pc)/Dy
name = "column"
EDP_threshold = [1,2,3,4]
EDP_std = [0.25,0.25,0.47,0.47]
dr = np.array([0.03,0.08,0.25,1.0]).reshape([1,4]) #damage ratio
fragility(Sa,true_EDP,sample_EDP,name,EDP_threshold,EDP_std,dr,ref_EDP,hazard)


true_EDP = np.max(np.abs(RESTH[:,:,2]),axis=1)
ref_EDP = np.max(np.abs(pred_lstm[:,:,2]),axis=1)
sample_EDP = np.array(Pb)
name = "bearing"
EDP_threshold = [2.54,10.16]
EDP_std = [0.25,0.25]
dr = np.array([0.5,1.0]).reshape([1,2]) #damage ratio
fragility(Sa,true_EDP,sample_EDP,name,EDP_threshold,EDP_std,dr,ref_EDP,hazard)



true_EDP = np.max(np.abs(RESTH[:,:,0]),axis=1)/Dy\
    +0.05*np.trapz(RESTH[:,:,1],-RESTH[:,:,0])/(Fy*Dy)
ref_EDP = np.max(np.abs(pred_lstm[:,:,0]),axis=1)/Dy\
    +0.05*np.trapz(pred_lstm[:,:,1],-pred_lstm[:,:,0])/(Fy*Dy)
sample_EDP = np.array(Pc)/Dy+0.05*np.array(Ec)/(Fy*Dy)
name='parker'
EDP_threshold = [1,2,3,4]
EDP_std = [0.25,0.25,0.47,0.47]
dr = np.array([0.03,0.08,0.25,1.0]).reshape([1,4]) #damage ratio
fragility(Sa,true_EDP,sample_EDP,name,EDP_threshold,EDP_std,dr,ref_EDP,hazard)

