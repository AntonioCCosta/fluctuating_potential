import numpy as np
import numpy.ma as ma
import argparse
import sys
import h5py
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

#change path_to_data to the data path

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


def main():
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

    dx=.01
    stride = int(delay)
    h=.08
    wsize = int(5*60*frameRate)
    n_sims=1000
    eps_sim = np.zeros((n_sims,wsize-stride))
    for ks in range(n_sims):
        kw = np.random.randint(12)
        X = ctraj_w[kw]
        edges_kernel = np.arange(X.min(), X.max(), dx)
        centers_kernel = 0.5*(edges_kernel[1:]+edges_kernel[:-1])
        f_kernel,a_kernel,f_err_kernel,a_err_kernel = fit_stochastic_model_kernel(X,centers_kernel,stride,data_dt,h)
        sel = ~np.isnan(f_kernel)
        spl_f_kernel = UnivariateSpline(centers_kernel[sel],f_kernel[sel], w=1/f_err_kernel[sel],k=1,s=0,ext='zeros')
        spl_a_kernel = UnivariateSpline(centers_kernel[sel],a_kernel[sel], w=1/a_err_kernel[sel],k=1,s=0,ext='zeros')
        xmin = centers_kernel[spl_f_kernel(centers_kernel)!=0][0]
        xmax = centers_kernel[spl_f_kernel(centers_kernel)!=0][-1]
        sel=np.all([~X[t:t+wsize].mask for t in range(len(X)-wsize)],axis=1)
        indices = np.arange(len(X)-wsize)[sel]
        print(indices.shape)
        t0 = np.random.choice(indices)
        sampleX = X[t0:t0+wsize]
        trange = np.arange(len(sampleX)-delay)
        for kt,t in enumerate(trange):
            x = sampleX[t]
            if x<xmin:
                D1 = spl_f_kernel(xmin)
                D2 = spl_a_kernel(xmin)
            elif x>xmax:
                D1 = spl_f_kernel(xmax)
                D2 = spl_a_kernel(xmax)
            else:
                D1 = spl_f_kernel(x)
                D2 = spl_a_kernel(x)
            xpred = x+D1*stride*data_dt
            xfut = sampleX[t+stride]
            eps_sim[ks,kt] = (xfut-xpred)/np.sqrt(2*D2)
        print(ks,kw,t0,flush=True)


    print('Saving results',flush=True)
    f = h5py.File('path_to_data/worm_analysis/noise_correlations/eps_sims.h5','w')
    eps_ = f.create_dataset('eps_sim',eps_sim.shape)
    eps_[...] = eps_sim
    ws_ = f.create_dataset('wsize',(1,))
    ws_[...] = wsize
    h_ = f.create_dataset('h',(1,))
    h_[...] = h
    dx_ = f.create_dataset('dx',(1,))
    dx_[...] = dx
    stride_ = f.create_dataset('stride',(1,))
    stride_[...] = stride
    f.close()


if __name__ == "__main__":
    main()
