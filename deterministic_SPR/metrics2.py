#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def cosSM(pred,truth,bound=0.8,color='red',axs=None):
    norm1=np.sqrt(np.sum(pred**2,axis=1))
    norm2=np.sqrt(np.sum(np.array(truth)**2,axis=1))
    r = np.sum(pred*np.array(truth),axis=1)/norm1/norm2
    if axs != None:
        xlim = [bound,1]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(r,rwidth=0.6,bins=bins,color=color,edgecolor='black', linewidth=1.2,density=True)
        axs.set_xlabel('Cosine similarity')
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,(xlim[-1]-bound)/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Density of instances')
        # axs.grid(visuable)
    return r

def Cal_cdf(data):
    #input is a vector
    n = len(data)
    x = np.sort(data) #x axis for cdf
    y = np.arange(1,n+1)/n #y axis for cdf
    return x,y

def Plot_cdf(data,axs,color,xlabel,label,visuable=None):
    x,y = Cal_cdf(data)
    axs.step(x,y,color=color,linewidth=2,label=label)
    axs.grid(visuable)
    axs.set_ylim(0,1.02)
    axs.set_yticks(np.arange(0,1.1,0.2))
    axs.set_xlim(-0.01,0.51)
    axs.set_xticks(np.arange(0,0.6,0.1))
    axs.set_ylabel('CDF')
    axs.set_xlabel(xlabel)

def Cal_Residual(pred,truth,axs=None,bound=1):
    #pred is matrix, truth is list
    threshold=0.05*pred.max()
    truth = np.array([ii[-100:] for ii in truth])+threshold
    pred = pred[:,-100:]+threshold
    nn = pred.shape[1]
    nume = abs(truth-pred)
    deno = abs(pred)+abs(truth)
    loss = (nume/deno).sum(axis=1)/nn
    if axs!=None:            
        x,y = Cal_cdf(loss)
        ind = np.argwhere(y>=bound)[0][0]
        xlim = [0,round(x[ind]*100)/100]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(loss,rwidth=0.6,bins=bins,color='orange',edgecolor='black', linewidth=1.2)
        axs.set_xlabel('Residual deformation')
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,xlim[-1]/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Number of instances')
        axs.grid()
    return loss

def Cal_Peak(pred,truth,axs=None,bound=1):
    true_P=np.array([max(abs(ii)) for ii in truth]).reshape(-1)
    pred_P=np.max(abs(pred[:,:]),axis=1)
    loss=abs(true_P-pred_P)/(0.1*true_P+pred_P)
    if axs!=None:
        x,y = Cal_cdf(loss)
        ind = np.argwhere(y>=bound)[0][0]
        xlim = [0,round(x[ind]*100)/100]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(loss,rwidth=0.6,bins=bins,color='orange',edgecolor='black', linewidth=1.2)
        axs.set_xlabel('Peak response')
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,xlim[-1]/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Number of instances')
        axs.grid()
    return loss

def Cal_Amp(pred,truth,axs=None,bound=1):
    #pred is matrix, truth is list
    nn = pred.shape[1]
    delta = abs(pred-np.array(truth))
    deno = abs(pred)+0.1*np.max(np.abs(truth))
    loss = (delta/deno).sum(axis=1)/nn
    if axs!=None:
        x,y = Cal_cdf(loss)
        ind = np.argwhere(y>=bound)[0][0]
        xlim = [0,round(x[ind]*100)/100]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(loss,rwidth=0.6,bins=bins,color='orange',edgecolor='black', linewidth=1.2)
        axs.set_xlabel('SMAPE') #Symmetric Mean Absolute Percentage Error
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,xlim[-1]/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Number of instances')
        axs.grid()
    return loss

def Cal_Eng(pred,truth,axs=None,bound=1):   
    predEng = np.sum(np.abs(pred),axis=1)
    predEng = predEng.reshape(-1)
    trueEng = np.array([sum(abs(ii)) for ii in truth]).reshape(-1)
    deno = predEng+0.1*trueEng
    #deno = np.maximum(predEng,trueEng).reshape(-1)
    Eng = np.abs(predEng-trueEng)/deno
    if axs!=None:
        x,y = Cal_cdf(Eng)
        ind = np.argwhere(y>=bound)[0][0]
        xlim = [0,round(x[ind]*10)/10.]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(Eng,rwidth=0.6,bins=bins,color='orange',edgecolor='black', linewidth=1.2)
        axs.set_xlabel('$\cal{E}$')
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,xlim[-1]/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Number of instances')
        axs.grid()
    return Eng

def Cal_correlation(pred,truth,color='orange',axs=None,bound=0,visuable=None):
    r=[np.corrcoef(truth[ii],pred[ii,:])[0,1] for ii in np.arange(len(truth))]
    if axs != None:
        xlim = [bound,1]
        bins = np.linspace(xlim[0],xlim[-1],8)
        axs.hist(r,rwidth=0.6,bins=bins,color=color,edgecolor='black', linewidth=1.2,density=True)
        axs.set_xlabel('$\cal{R}$')
        axs.set_xticks(np.arange(0,xlim[-1]*1.1,(xlim[-1]-bound)/5))
        axs.set_xlim(xlim)
        axs.set_ylabel('Density of instances')
        axs.grid(visuable)
    return r

def Plot_correlation(x,axs,color,bound=0.6,label=['Train','Test']):
    xlim = [bound,1]
    bins = np.linspace(xlim[0],xlim[-1],8)
    axs.hist(x, bins, histtype='bar',color=color,edgecolor=None,density=False,label=label)
    axs.set_xlabel('${\cal{R}}^2$')
    axs.set_xticks(np.arange(0,xlim[-1]*1.1,(xlim[-1]-bound)/4))
    axs.set_xlim(xlim)
    axs.set_ylabel('Number of instances')
    axs.grid()

def Cal_Dist(pred,truth,bound=.99,axs=None,color='r',leg=False,visuable=False):
    norm = np.array([np.max(ii) for ii in truth]).reshape((1,-1))
    loss = pred[:,:]-np.array(truth)
    norm_error = np.transpose(np.transpose(loss)/norm)
    mu, std = np.zeros((len(truth),)),np.zeros((len(truth),))
    for ii in np.arange(len(truth)):
        mu[ii], std[ii] = stats.norm.fit(norm_error[ii,:])
    #loss = Cal_Eng(pred,truth)
    if axs!=None:
        r=Cal_correlation(pred,truth)
        rank = np.argsort(r)
        i0 = rank[int((bound)*len(r))]
        xx = np.maximum(abs(mu[i0]-3*std[i0]),abs(mu[i0]+3*std[i0]))
        xx = np.round(xx*50)/50
        xlim = [-xx,xx]
        #xlim = [np.round((-3*std[i0])*10)/10,np.round((+3*std[i0])*10)/10]
        x00 = np.linspace(xlim[0],xlim[-1], 100)
        p1 = stats.norm.pdf(x00, mu[i0], std[i0])        
        axs.plot(x00,p1,color=color,linewidth=2)
        if leg:
            text = "%.2f confidence"%(1-bound)
            axs.text(x00[1], 1.1*max(p1),text,fontsize=12)
        # axs.set_ylim(-0.01,1.1*max(p1))
        # axs.set_xlim(xlim)
        # axs.set_xticks(np.linspace(xlim[0],xlim[-1],5))
        axs.set_xlabel('$\cal{N}$')
        axs.set_ylabel('Density')
        axs.grid(visuable)
    return mu[i0],std[i0]