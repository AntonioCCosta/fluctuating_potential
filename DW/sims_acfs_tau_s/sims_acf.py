import numpy as np
import argparse
import sys
import time
import h5py
#change path_to_utils and path_to_data accordingly
sys.path.append("path_to_utils/")
import stats

import numba
from numba import jit,prange,objmode
@jit(nopython=True,parallel=True)
def simulate_DW_acf(x0s,s0s,dt,stride,tau_s,mu_s,Tx,Ts,max_iters,lags):
    def nc_acf(x, lags=500):
        C_nc = np.zeros((len(lags),))
        C = np.zeros((len(lags),))
        X = x-x.mean()
        sigma2 = X.var()
        for i, l in enumerate(lags):
            if l == 0:
                C_nc[i] = (x*x).mean()
                C[i] = 1
            else:
                C_nc[i] = (x[:-l]*x[l:]).mean()
                C[i] = (X[:-l]*X[l:]).mean()/sigma2
        return C,C_nc
    n_sims = len(x0s)
    C_nc_sims = np.zeros((n_sims,len(lags)))
    C_sims = np.zeros((n_sims,len(lags)))
    for ks in prange(n_sims):
        x0 = x0s[ks]
        s0 = s0s[ks]
        x=x0
        s=s0
        xsim = np.zeros(int(max_iters/stride))
        xsim[0]=x0
        k=0
        for i in range(max_iters-1):
            new_x = x - 4*s**2*x*(x**2-1)*dt + np.sqrt(2*Tx)*np.random.normal(0, np.sqrt(dt))
            new_s = s - (1/tau_s)*(s-mu_s)*dt + np.sqrt(2*Ts*(1/tau_s))*np.random.normal(0, np.sqrt(dt))
            x = new_x
            s = new_s
            if i%stride==0:
                xsim[k+1] = x
                k+=1
        with objmode():
            print(ks,xsim.mean(),flush=True)
        acf,nc_cf = nc_acf(xsim,lags)
        C_sims[ks] = acf
        C_nc_sims[ks] = nc_cf
    return C_sims,C_nc_sims

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)
    Tx,Ts,eps_s = np.loadtxt('iteration_Ts_eps.txt')[idx]

    n_sims=50000
    mu_s = np.sqrt(Tx)
    print(Tx,Ts,flush=True)
    s0s = np.random.normal(0,np.sqrt(Ts),n_sims)
    x0s = np.random.normal(0,np.sqrt(Tx),n_sims)
    iter_dt=2e-1
    dt = 100
    stride = int(dt/iter_dt)
    maxT = int(1e8)
    tau_s = maxT*eps_s
    max_iters = int(maxT/iter_dt)
    nlags=100
    lags = np.unique(np.array(np.logspace(np.log10(int(5e4/dt)),np.log10(int(max_iters/stride)-1),nlags),dtype=int))
    C_sims,C_nc_sims = simulate_DW_acf(x0s,s0s,iter_dt,stride,tau_s,mu_s,Tx,Ts,max_iters,lags)
    print(C_sims.shape,flush=True)


    f = h5py.File('path_to_data/DW/sims_acfs_tau_s/acfs_{}.h5'.format(idx),'w')
    mD = f.create_group('metaData')
    Tx_ = mD.create_dataset('Tx',(1,))
    Tx_[...] = Tx
    Ts_ = mD.create_dataset('Ts',(1,))
    Ts_[...] = Ts
    eps_ = mD.create_dataset('eps_s',(1,))
    eps_[...] = eps_s
    dt_ = mD.create_dataset('dt',(1,))
    dt_[...] = dt
    iter_dt_ = mD.create_dataset('iter_dt',(1,))
    iter_dt_[...] = iter_dt
    maxT_ = mD.create_dataset('maxT',(1,))
    maxT_[...] = maxT
    C_ = f.create_dataset('C_sims',C_sims.shape)
    C_[...] = C_sims
    Cnc_ = f.create_dataset('C_nc_sims',C_nc_sims.shape)
    Cnc_[...] = C_nc_sims
    l_ = f.create_dataset('lags',lags.shape)
    l_[...] = lags
    f.close()

    print('Saved results',flush=True)

if __name__ == "__main__":
    main(sys.argv)
