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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)

    h_range = np.arange(0.01,.5,.01)

    frameRate=16.
    data_dt=1/frameRate

    ctraj_path = 'path_to_data/worm_analysis/ctrajs_1000_clusters/' #needs to be redefined
    f = h5py.File(ctraj_path+'/c_traj_w.h5','r')
    mD = f['MetaData']
    n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
    delay = np.array(mD['delay'],dtype=int)[0]
    ctraj_w = ma.array(f['ctraj_w'])
    ctraj_w_mask = ma.array(f['ctraj_w_mask'])
    f.close()
    ctraj_w[ctraj_w_mask==1]=ma.masked

    dx = .01
    stride = int(delay)

    kw = np.random.randint(0,len(ctraj_w))
    X = ctraj_w[kw]
    edges_kernel = np.arange(X.min(), X.max(), dx)
    centers_kernel = 0.5*(edges_kernel[1:]+edges_kernel[:-1])
    delta_h = np.zeros((len(h_range),2))
    for kh,h in enumerate(h_range):
        f_kernel,a_kernel,f_err_kernel,a_err_kernel = fit_stochastic_model_kernel(X,centers_kernel,stride,data_dt,h)
        sel = ~np.isnan(f_kernel)
        spl_f_kernel = UnivariateSpline(centers_kernel[sel],f_kernel[sel], w=1/f_err_kernel[sel],k=1,s=0,ext='zeros')
        spl_a_kernel = UnivariateSpline(centers_kernel[sel],a_kernel[sel], w=1/a_err_kernel[sel],k=1,s=0,ext='zeros')
        xmin = centers_kernel[spl_f_kernel(centers_kernel)!=0][0]
        xmax = centers_kernel[spl_f_kernel(centers_kernel)!=0][-1]
        x0 = X.compressed()[0]
        maxT = int(len(X)*data_dt)
        max_iters = int(maxT/data_dt)
        xsim = simulate_inferred_model(x0,data_dt,spl_f_kernel,spl_a_kernel,xmin,xmax,max_iters)
        edges_rec = np.arange(xsim.min(), xsim.max(), dx)
        centers_rec = 0.5*(edges_rec[1:]+edges_rec[:-1])
        f_rec,a_rec,f_err_rec,a_err_rec = fit_stochastic_model_kernel(xsim,centers_rec,stride,data_dt,h)
        f_hat,a_hat,f_err_hat,a_err_hat = fit_stochastic_model_kernel(X,centers_rec,stride,data_dt,h)
        sel = ~np.isnan(f_rec)
        pB_rec = np.zeros(centers_rec.shape)
        pB_rec[sel] = np.exp(np.cumsum(f_rec[sel]/a_rec[sel]*dx))/a_rec[sel]
        pB_rec = pB_rec/np.trapz(pB_rec,centers_rec)
        sel = ~np.isnan(f_hat)
        pB_hat = np.zeros(centers_rec.shape)
        pB_hat[sel] = np.exp(np.cumsum(f_hat[sel]/a_hat[sel]*dx))/a_hat[sel]
        pB_hat = pB_hat/np.trapz(pB_hat,centers_rec)
        h_D1 = np.nansum(np.abs(f_hat-f_rec)*np.sqrt(pB_hat*pB_rec)*dx)/np.nansum(np.sqrt(pB_hat*pB_rec)*dx)
        h_D2 = np.nansum(np.abs(a_hat-a_rec)*np.sqrt(pB_hat*pB_rec)*dx)/np.nansum(np.sqrt(pB_hat*pB_rec)*dx)
        delta_h[kh] = [h_D1,h_D2]
        print(h,kw,h_D1,h_D2,flush=True)



    print('Saving results',flush=True)
    f = h5py.File('path_to_data/worm_analysis/optimize_h/delta_h_{}.h5'.format(idx),'w')
    dh_ = f.create_dataset('delta_h',delta_h.shape)
    dh_[...] = delta_h
    h_ = f.create_dataset('h_range',h_range.shape)
    h_[...] = h_range
    dx_ = f.create_dataset('dx',(1,))
    dx_[...] = dx
    stride_ = f.create_dataset('stride',(1,))
    stride_[...] = stride
    kw_ = f.create_dataset('kw',(1,))
    kw_[...] = kw
    f.close()


if __name__ == "__main__":
    main(sys.argv)
