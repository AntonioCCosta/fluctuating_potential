import numpy as np
import numpy.ma as ma
import argparse
import sys
sys.path.append("/home/a/antonio-costa/BehaviorModel/utils/")
import operator_calculations as op_calc
import delay_embedding as embed
import stats
from scipy.interpolate import UnivariateSpline
from multiprocessing import Pool,cpu_count
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data_dt=1
delay=3

ctraj_path = '/flash/StephensU/antonio/BehaviorModel/Zebrafish/ctrajs_562_clusters_delay_{}/'.format(delay)
f = h5py.File(ctraj_path+'/c_traj_fish.h5','r')
mD = f['MetaData']
n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
delay = np.array(mD['delay'],dtype=int)[0]
ctraj_fish = ma.array(f['ctraj_fish'])
ctraj_fish_mask = ma.array(f['ctraj_fish_mask'])
f.close()
ctraj_fish[ctraj_fish_mask==1]=ma.masked

sims_tv_fish=[]
sims_fish=[]
dts_sims_fish=[]
dts_sims_tv_fish=[]
for kf in range(len(ctraj_fish)):
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/Zebrafish/sims_fpts/sims_fish_{}.h5'.format(kf),'r')
    sims = np.array(f['sims'])
    sims_tv = np.array(f['sims_tv'])
    dts_sims = np.array(f['dts_sims'])
    dts_sims_tv = np.array(f['dts_sims_tv'])
    dx = np.array(f['metaData/dx'])[0]
    h = np.array(f['metaData/h'])[0]
    overlap = np.array(f['metaData/overlap'])[0]
    step = np.array(f['metaData/step'],dtype=int)[0]
    stride = np.array(f['metaData/stride'],dtype=int)[0]
    wsize = np.array(f['metaData/wsize'],dtype=int)[0]
    f.close()
    sims_fish.append(sims)
    sims_tv_fish.append(sims_tv)
    dts_sims_fish.append(dts_sims)
    dts_sims_tv_fish.append(dts_sims_tv)
    print(kf)
    
    
def get_errorbar_dist(lifetimes_w,t0,tf):
    all_lt = np.hstack(lifetimes_w)
    x,y = stats.complementary_cumulative_dist(all_lt,(t0,tf))
    y_all = np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
    x_all = np.sort(np.unique(x))
    
    dict_y = {}
    for x_ in x_all:
        dict_y[x_] = []

    for k in range(100):
        x,y = stats.cumulative_dist(np.hstack(np.random.choice(lifetimes_w,len(lifetimes_w))),(t0,tf))
        y = 1-np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
        x = np.sort(np.unique(x))
        for kx in range(len(y)):
            dict_y[x[kx]].append(y[kx])
        if k%10==0:
            print(k)
            
    y_errorbars = np.zeros((len(dict_y.keys()),3))
    for kx,x_ in enumerate(x_all):
        values = np.array(dict_y[x_])
        values = values[values>0]
        cil = np.percentile(values,2.5)
        ciu = np.percentile(values,97.5)
        y_errorbars[kx] = [y_all[kx],cil,ciu]
    return x_all,y_errorbars



x_sim,y_err_sim = get_errorbar_dist(dts_sims_fish,0,10000)

x_sim_tv,y_err_sim_tv = get_errorbar_dist(dts_sims_tv_fish,0,10000)

def state_lifetime(states):
    durations=[]
    for state in np.sort(np.unique(states.compressed())):
        gaps = states==state
        gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
        durations.append(np.hstack(np.diff(gaps_boundaries)))
    return durations


dts_fish=[]
for kw,X in enumerate(ctraj_fish):
    labels = ma.zeros(X.shape,dtype=int)
    labels[X>0] = 1
    labels[X.mask] = ma.masked
    dts = np.hstack(state_lifetime(labels[::stride]))*stride*data_dt
    dts_fish.append(dts)
    
    
x_data,y_err_data = get_errorbar_dist(dts_fish,0,10000)

f=h5py.File('fpts_errbars.h5','w')
xsim = f.create_dataset('x_sim',x_sim.shape)
xsim[...] = x_sim
ysim = f.create_dataset('y_err_sim',y_err_sim.shape)
ysim[...] = y_err_sim
xsimtv = f.create_dataset('x_sim_tv',x_sim_tv.shape)
xsimtv[...] = x_sim_tv
ysimtv = f.create_dataset('y_err_sim_tv',y_err_sim_tv.shape)
ysimtv[...] = y_err_sim_tv
xdata = f.create_dataset('x_data',x_data.shape)
xdata[...] = x_data
ydata= f.create_dataset('y_err_data',y_err_data.shape)
ydata[...] = y_err_data
f.close()