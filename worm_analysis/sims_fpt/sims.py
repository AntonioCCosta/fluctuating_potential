import numpy as np
import numpy.ma as ma
import argparse
import sys
import h5py
from scipy.interpolate import UnivariateSpline


def kernel(x):
    kernel_x = ma.zeros(x.shape)
    sel = x**2<5
    kernel_x[sel] = (3/(4*np.sqrt(5)))*(1-x[sel]**2/5)
    return kernel_x

def K(h,x):
    return (1/h)*kernel(x/h)

def W(h,j,x,X):
    kernel_all = K(h,x-X)
    num = kernel_all[j]
    den = kernel_all.mean()
    return num/den

def fit_sde_errorbars(x,X,stride,data_dt,h):
    tau=stride*data_dt
    X_lag = ma.vstack([X[:-stride],X[stride:]]).T
    sel = ~np.any(X_lag.mask,axis=1)
    indices = np.arange(X[:-stride].shape[0])[sel]
    N = sel.sum()
    #W function
    kernel_all = K(h,x-X)
    kernel_mean = kernel_all.mean()
    terms = D1_D2_parallel(indices,kernel_all,kernel_mean,X,stride)
    D1_mean,D2_mean = np.mean(terms,axis=0)
    D1 = D1_mean/tau
    D2 = D2_mean/(2*tau)
    D1_std,D2_std = np.std(terms,axis=0)
    D1_err = D1_std/tau/np.sqrt(len(terms))
    D2_err = D2_std/(2*tau)/np.sqrt(len(terms))
    return D1,D1_err,D2,D2_err

from numba import jit,prange
@jit(nopython=True,parallel=True)
def D1_D2_parallel(indices,kernel_all,kernel_mean,X,stride):
    terms = np.zeros((len(indices),2))
    for kj in prange(len(indices)):
        j = indices[kj]
        t1 = (kernel_all[j]/kernel_mean)*(X[j+stride]-X[j])
        t2 = (kernel_all[j]/kernel_mean)*(X[j+stride]-X[j])**2
        terms[kj] = [t1,t2]
    return terms

def fit_stochastic_model_kernel(X,centers_kernel,stride,data_dt,h):
    f_kernel = np.zeros(len(centers_kernel))
    a_kernel = np.zeros(len(centers_kernel))
    f_err_kernel = np.zeros(len(centers_kernel))
    a_err_kernel = np.zeros(len(centers_kernel))
    for kc,x in enumerate(centers_kernel):
        D1,D1_err,D2,D2_err = fit_sde_errorbars(x,X,stride,data_dt,h)
        f_kernel[kc] = D1
        a_kernel[kc] = D2
        f_err_kernel[kc] = D1_err
        a_err_kernel[kc] = D2_err
    return f_kernel,a_kernel,f_err_kernel,a_err_kernel


def simulate_inferred_model(x0,dt,spl_f,spl_a,xmin,xmax,max_iters):
    xsim = np.zeros(max_iters)
    x=x0
    for i in range(max_iters):
        xsim[i] = x
        if x<xmin:
            D1 = spl_f(xmin)
            D2 = spl_a(xmin)
        elif x>xmax:
            D1 = spl_f(xmax)
            D2 = spl_a(xmax)
        else:
            D1 = spl_f(x)
            D2 = spl_a(x)
        new_x = x + D1*dt + np.sqrt(2*D2)*np.random.normal(0, np.sqrt(dt))
        x = new_x
    return xsim

def simulate_tv_inferred_model(x0,dt,centers,spl_f_t,spl_a_t,twindows,max_iters):
    xsim = np.zeros(max_iters)
    x=x0
    for i in range(max_iters):
        t_idx = np.argmin(np.abs(twindows-i*dt))
        spl_f = spl_f_t[t_idx]
        spl_a = spl_a_t[t_idx]
        xmin = centers[spl_f(centers)!=0][0]
        xmax = centers[spl_f(centers)!=0][-1]
        xsim[i] = x
        if x<xmin:
            D1 = spl_f(xmin)
            D2 = spl_a(xmin)
        elif x>xmax:
            D1 = spl_f(xmax)
            D2 = spl_a(xmax)
        else:
            D1 = spl_f(x)
            D2 = spl_a(x)
        new_x = x + D1*dt + np.sqrt(2*D2)*np.random.normal(0, np.sqrt(dt))
        x = new_x
    return xsim

def state_lifetime(states):
    durations=[]
    for state in np.sort(np.unique(states.compressed())):
        gaps = states==state
        gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
        durations.append(np.hstack(np.diff(gaps_boundaries)))
    return durations

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm',help='index',default=0,type=int)
    args=parser.parse_args()
    kw = int(args.worm)
    print(kw,flush=True)
    print('Load data',flush=True)

    frameRate=16.
    data_dt=1/frameRate


    ctraj_path = 'path_to_data/worm_analysis/ctrajs_1000_clusters/'
    f = h5py.File(ctraj_path+'/c_traj_w.h5','r')
    mD = f['MetaData']
    n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
    delay = np.array(mD['delay'],dtype=int)[0]
    ctraj_w = ma.array(f['ctraj_w'])
    ctraj_w_mask = ma.array(f['ctraj_w_mask'])
    f.close()
    ctraj_w[ctraj_w_mask==1]=ma.masked

    X = ctraj_w[kw]
    maxT = int(X.shape[0]*data_dt)
    max_iters = int(maxT/data_dt)
    ts_data = np.arange(0, maxT, data_dt)
    stride = delay
    step = int(0.5*delay)
    h=.08
    dx = 1e-2
    n_sims = 5000

    wsize = int(300/data_dt)
    overlap = .1

    edges_kernel = np.arange(X.min(), X.max(), dx)
    centers_kernel = 0.5*(edges_kernel[1:]+edges_kernel[:-1])
    f_kernel,a_kernel,f_err_kernel,a_err_kernel = fit_stochastic_model_kernel(X,centers_kernel,stride,data_dt,h)
    sel = ~np.isnan(f_kernel)
    spl_f_kernel = UnivariateSpline(centers_kernel[sel],f_kernel[sel], w=1/f_err_kernel[sel],k=1,s=0,ext='zeros')
    spl_a_kernel = UnivariateSpline(centers_kernel[sel],a_kernel[sel], w=1/a_err_kernel[sel],k=1,s=0,ext='zeros')

    xmin = centers_kernel[spl_f_kernel(centers_kernel)!=0][0]
    xmax = centers_kernel[spl_f_kernel(centers_kernel)!=0][-1]
    x0 = X.compressed()[0]

    sims=np.zeros((n_sims,max_iters))
    for k in range(n_sims):
        sim = simulate_inferred_model(x0,data_dt,spl_f_kernel,spl_a_kernel,xmin,xmax,max_iters)
        sims[k]= sim
        print(kw,k,flush=True)

    dts_sims=[]
    for sim in sims:
        labels_sim = ma.zeros(sim.shape,dtype=int)
        labels_sim[sim>0] = 1
        dts_sim = np.hstack(state_lifetime(labels_sim[::step]))*step*data_dt
        dts_sims.append(dts_sim)

    wstarts = np.arange(0,len(X)-wsize,int(wsize*overlap))
    spl_f_kernel_t = []
    spl_a_kernel_t = []
    indices=[]
    for kt,t0 in enumerate(wstarts):
        if ma.count_masked(X[t0:t0+wsize])<0.2*wsize:
            indices.append(kt)
            f_kernel,a_kernel,f_err_kernel,a_err_kernel = fit_stochastic_model_kernel(X[t0:t0+wsize],centers_kernel,stride,data_dt,h)
            sel = ~np.logical_or(np.isnan(f_kernel),f_kernel==0)
            spl_f_ = UnivariateSpline(centers_kernel[sel],f_kernel[sel], w=1/f_err_kernel[sel],k=1,s=0,ext='zeros')
            spl_a_ = UnivariateSpline(centers_kernel[sel],a_kernel[sel], w=1/a_err_kernel[sel],k=1,s=0,ext='zeros')
            spl_f_kernel_t.append(spl_f_)
            spl_a_kernel_t.append(spl_a_)
    twindows = (wstarts[indices]+wsize/2)*data_dt

    sims_tv=np.zeros((n_sims,max_iters))
    for k in range(n_sims):
        sim_tv = simulate_tv_inferred_model(x0,data_dt,centers_kernel,spl_f_kernel_t,spl_a_kernel_t,twindows,max_iters)
        sims_tv[k] = sim_tv
        print(kw,k,flush=True)

    dts_sims_tv=[]
    for sim_tv in sims_tv:
        labels_sim_tv = ma.zeros(sim_tv.shape,dtype=int)
        labels_sim_tv[sim_tv>0] = 1
        dts_sim_tv = np.hstack(state_lifetime(labels_sim_tv[::step]))*step*data_dt
        dts_sims_tv.append(dts_sim_tv)


    print('Saving results',flush=True)
    f = h5py.File('path_to_data/worm_analysis/sims_fpts/sims_w_{}.h5'.format(kw),'w')
    xs_ = f.create_dataset('sims',sims.shape)
    xs_[...] = sims
    xs_tv_ = f.create_dataset('sims_tv',sims_tv.shape)
    xs_tv_[...] = sims_tv
    dts_ = f.create_dataset('dts_sims',np.hstack(dts_sims).shape)
    dts_[...] = np.hstack(dts_sims)
    dts_tv_ = f.create_dataset('dts_sims_tv',np.hstack(dts_sims_tv).shape)
    dts_tv_[...] = np.hstack(dts_sims_tv)
    mD = f.create_group('metaData')
    h_ = mD.create_dataset('h',(1,))
    h_[...] = h
    dx_ = mD.create_dataset('dx',(1,))
    dx_[...] = dx
    st_ = mD.create_dataset('stride',(1,))
    st_[...] = stride
    dt_ = mD.create_dataset('data_dt',(1,))
    dt_[...] = data_dt
    step_ = mD.create_dataset('step',(1,))
    step_[...] = step
    ws_ = mD.create_dataset('wsize',(1,))
    ws_[...] = wsize
    ol_ = mD.create_dataset('overlap',(1,))
    ol_[...] = overlap
    f.close()


if __name__ == "__main__":
    main(sys.argv)
