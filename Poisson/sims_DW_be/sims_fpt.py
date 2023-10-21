import numpy as np
import argparse
import sys
import time
import h5py
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

#change path_to_data to the data path


def potential(x,s):
    return s**2*(x**2-1)**2

def exppot(x, Tx,s,sign=-1, fun=lambda z: 1):
    return np.exp(sign*potential(x,s)/Tx)*fun(x)

def omega_theory(Tx,s):
    z = np.linspace(-20, 0, 1000)
    iarr = cumtrapz(exppot(z,Tx,s), z, initial=0)
    ifun = interp1d(z, iarr)

    y = np.linspace(-1, 0, 1000)
    oarr = cumtrapz(exppot(y,Tx,s, sign=1, fun=ifun), y, initial=0)/Tx
    ofun = interp1d(y, oarr)

    return 1/(2*ofun(0))

from numba import jit,prange
import numba
@jit(nopython=True)
def Poisson_fpt(s_sample,omega_sample,maxT,n_sims,n_iters_max = 10000000):

    dts=[]
    omegas = []
    s_sims=[]
    for kw in prange(len(omega_sample)):
        omega = omega_sample[kw]
        t=0
        for k in range(0,n_iters_max):
            step = np.random.exponential(1/omega, 1)[0]
            t=t+step
            if t<=maxT:
                dts.append(step)
                omegas.append(omega)
                s_sims.append(s_sample[kw])
            if t>maxT:
                break

    return np.array(s_sims),np.array(omegas),np.array(dts)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)
    ratio = np.linspace(.25,2,11)

    Tx=1e-3
    mu_s = 1*np.sqrt(Tx)
    maxT = 1e7

    Ts = Tx*ratio[idx]

    n_sims = 50000

    s_samples = np.random.normal(mu_s,np.sqrt(Ts),n_sims)
    omega_samples = np.array([omega_theory(Tx,s) for s in s_samples])
    sel = omega_samples>0
    omega_sample = omega_samples[sel]
    s_sample = s_samples[sel]


    s_sim,omegas_sim,dts_sim = Poisson_fpt(s_sample,omega_sample,maxT,n_sims)
    s_sim = np.hstack(s_sim)
    omegas_sim=np.hstack(omegas_sim)
    dts_sim = np.hstack(dts_sim)

    print(dts_sim.shape,flush=True)

    f = h5py.File('path_to_data/Poisson/sims_fpts_be/dts_{}.h5'.format(idx),'w')
    Tx_ = f.create_dataset('Tx',(1,))
    Tx_[...] = Tx
    mu = f.create_dataset('mu_s',(1,))
    mu[...] = mu_s
    maxT_ = f.create_dataset('maxT',(1,))
    maxT_[...] = maxT
    s_ = f.create_dataset('s_sim',s_sim.shape)
    s_[...] = s_sim
    w_ = f.create_dataset('omegas_sim',omegas_sim.shape)
    w_[...] = omegas_sim
    t_ = f.create_dataset('dts_sim',dts_sim.shape)
    t_[...] = dts_sim
    f.close()

    print('Saved results')

if __name__ == "__main__":
    main(sys.argv)
