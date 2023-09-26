import numpy as np
import numpy.ma as ma
import argparse
import sys
sys.path.append("~/Repositories/fluctuating_potential_repo+data/fluctuating_potential/utils/")
import operator_calculations as op_calc
import delay_embedding as embed
import stats
import h5py

from lmfit import minimize, Parameters, Parameter, report_fit

def fit_tscales(x,y,b0=10,d0=50):

    def fcn2(params,x,data):
        a=params['a'].value
        b=params['b'].value
        c=params['c'].value
        d=params['d'].value
        #Fits the drift

        return (a*np.exp(-x/b) + c * np.exp(-x/d) - data)

    params = Parameters()
    params.add('a',   value= 1)
    params.add('b', value= b0,min=0)
    params.add('c', value= 1)
    params.add('d', value= d0,min=0)

    # do fit, here with leastsq model
    result = minimize(fcn2, params, args=(x,y))

    p1 = result.params['a'].value
    p2 = result.params['b'].value
    p3 = result.params['c'].value
    p4 = result.params['d'].value
    tscales = np.array([p2,p4])
    stderrs = np.array([result.params['b'].stderr,result.params['d'].stderr])

    return p1,p2,p3,p4,tscales,stderrs

def func2(x, a, b, c, d):
    return a * np.exp(-x/b) + c * np.exp(-x/d)

def state_lifetime(states):
    durations=[]
    for state in np.sort(np.unique(states.compressed())):
        gaps = states==state
        gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
        durations.append(np.hstack(np.diff(gaps_boundaries)))
    return durations

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

def main():
    print('Load data',flush=True)

    frameRate=16.
    data_dt=1/frameRate

    ctraj_path = '~/Repositories/fluctuating_potential_repo+data/data/worm_analysis/ctrajs_1000_clusters/'
    f = h5py.File(ctraj_path+'/c_traj_w.h5','r')
    mD = f['MetaData']
    n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
    delay = np.array(mD['delay'],dtype=int)[0]
    ctraj_w = ma.array(f['ctraj_w'])
    ctraj_w_mask = ma.array(f['ctraj_w_mask'])
    f.close()
    ctraj_w[ctraj_w_mask==1]=ma.masked

    dts_sims_w=[]
    dts_sims_tv_w=[]
    for kw in range(len(ctraj_w)):
        f = h5py.File('~/Repositories/fluctuating_potential_repo+data/data/worm_analysis/sims_fpts/sims_w_{}.h5'.format(kw),'r')
        dts_sims = np.array(f['dts_sims'])
        dts_sims_tv = np.array(f['dts_sims_tv'])
        dx = np.array(f['metaData/dx'])[0]
        h = np.array(f['metaData/h'])[0]
        overlap = np.array(f['metaData/overlap'])[0]
        step = np.array(f['metaData/step'],dtype=int)[0]
        stride = np.array(f['metaData/stride'],dtype=int)[0]
        wsize = np.array(f['metaData/wsize'],dtype=int)[0]
        f.close()
        dts_sims_w.append(dts_sims)
        dts_sims_tv_w.append(dts_sims_tv)
        print(kw)

    print('Estimate from data',flush=True)
    dts_w=[]
    for kw,X in enumerate(ctraj_w):
        labels = ma.zeros(X.shape,dtype=int)
        labels[X>0] = 1
        labels[X.mask] = ma.masked
        dts = np.hstack(state_lifetime(labels[::step]))*step*data_dt
        dts_w.append(dts)

    x_data,y_err_data = get_errorbar_dist(dts_w,0,1000)

    t0,tf = delay*data_dt*.5,1000
    all_lt = np.hstack(dts_w)
    x,y = stats.complementary_cumulative_dist(all_lt,(t0,tf))
    y_all = np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
    x_all = np.sort(np.unique(x))
    p1,p2,p3,p4,tscales,stderrs = fit_tscales(x_all,y_all,b0=1,d0=20)
    params_all = np.array([p1,p2,p3,p4])
    sorted_indices = np.argsort(tscales)[::-1]
    tscales_all = tscales[sorted_indices]

    dict_y = {}
    for x_ in x_all:
        dict_y[x_] = []

    tscales_bootstrap = []
    params_bootstrap = []
    rates_bootstrap = []
    for k in range(5000):
        x,y = stats.complementary_cumulative_dist(np.hstack(np.random.choice(dts_w,len(dts_w))),(t0,tf))
        y = np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
        x = np.sort(np.unique(x))
        p1,p2,p3,p4,tscales,stderrs = fit_tscales(x,y,b0=1,d0=25)
        params_bootstrap.append(np.array([p1,p2,p3,p4]))
        sorted_indices = np.argsort(tscales)[::-1]
        tscales = tscales[sorted_indices]
        tscales_bootstrap.append(tscales)
        rates_bootstrap.append((1/tscales).sum())
        for kx in range(len(y)):
            dict_y[x[kx]].append(y[kx])

    tscales_cil = np.percentile(np.vstack(tscales_bootstrap),2.5,axis=0)
    tscales_ciu = np.percentile(np.vstack(tscales_bootstrap),97.5,axis=0)

    p1,p2,p3,p4 = params_all
    curvey = func2(x_all,p1,p2,p3,p4)


    print('Estimate from Markov model',flush=True)

    f = h5py.File('~/Repositories/fluctuating_potential_repo+data/data/worm_data/labels_tree/labels_tree.h5','r')
    delay = int(np.array(f['delay'])[0])
    eigfunctions = np.array(f['eigfunctions'])
    final_labels = ma.masked_invalid(np.array(f['final_labels'],dtype=int))
    final_labels_mask = np.array(f['final_labels_mask'])
    sel = final_labels_mask==1
    final_labels[sel] = ma.masked
    labels_tree = np.array(f['labels_tree'])
    f.close()
    eigfunctions_traj = ma.array(eigfunctions)[final_labels,:]
    eigfunctions_traj[final_labels.mask] = ma.masked

    n_worms=12
    kmeans_labels = labels_tree[0,:]

    f = h5py.File('~/Repositories/fluctuating_potential_repo+data/data/worm_data/symbol_seq_sims/symbol_sequence_simulations.h5','r')
    sims_w = []
    for worm in range(n_worms):
        sims_w.append(np.array(f[str(worm)]['sims'],dtype=int))
    f.close()

    n_sims=1000
    dts_sims_Markov_w = []
    for kw in range(n_worms):
        dts_sims = []
        for ks in range(n_sims):
            cluster_traj_sim = ma.copy(sims_w[kw][ks])
            cluster_traj_sim = ma.array(kmeans_labels,dtype=int)[sims_w[kw][ks]]
            dts_sim = np.hstack(state_lifetime(cluster_traj_sim))*data_dt*delay
            dts_sims.append(dts_sim)
        dts_sims_Markov_w.append(np.hstack(dts_sims))

    x_sim_Markov,y_err_sim_Markov = get_errorbar_dist(dts_sims_Markov_w,0,1000)


    print('Estimate from sims',flush=True)

    x_sim,y_err_sim = get_errorbar_dist(dts_sims_w,0,1000)

    x_sim_tv,y_err_sim_tv = get_errorbar_dist(dts_sims_tv_w,0,1000)

    print('Saving results',flush=True)
    f = h5py.File('~/Repositories/fluctuating_potential_repo+data/data/worm_analysis/fpts_errorbars/fpts_errorbars.h5','w')
    xd_ = f.create_dataset('x_data',x_data[:-1].shape)
    xd_[...] = x_data[:-1]
    yd_ = f.create_dataset('y_err_data',y_err_data[:-1].shape)
    yd_[...] = y_err_data[:-1]
    xs_ = f.create_dataset('x_sim',x_sim[:-1].shape)
    xs_[...] = x_sim[:-1]
    ys_ = f.create_dataset('y_err_sim',y_err_sim[:-1].shape)
    ys_[...] = y_err_sim[:-1]
    xsM_ = f.create_dataset('x_sim_Markov',x_sim_Markov[:-1].shape)
    xsM_[...] = x_sim_Markov[:-1]
    ysM_ = f.create_dataset('y_err_sim_Markov',y_err_sim_Markov[:-1].shape)
    ysM_[...] = y_err_sim_Markov[:-1]
    xs_tv_ = f.create_dataset('x_sim_tv',x_sim_tv[:-1].shape)
    xs_tv_[...] = x_sim_tv[:-1]
    ys_tv_ = f.create_dataset('y_err_sim_tv',y_err_sim_tv[:-1].shape)
    ys_tv_[...] = y_err_sim_tv[:-1]
    xa_ = f.create_dataset('x_all',x_all.shape)
    xa_[...] = x_all
    cy_ = f.create_dataset('curvey',curvey.shape)
    cy_[...] = curvey
    tsa_ = f.create_dataset('tscales_all',tscales_all.shape)
    tsa_[...] = tscales_all
    tscil_ = f.create_dataset('tscales_cil',tscales_cil.shape)
    tscil_[...] = tscales_cil
    tsciu_ = f.create_dataset('tscales_ciu',tscales_ciu.shape)
    tsciu_[...] = tscales_ciu
    f.close()


if __name__ == "__main__":
    main()
