import numpy as np
from numba import jit,prange
import numba
import argparse
import sys
import time
import h5py
@jit(nopython=True)
def integrate(h,Tl,Tx,k,n,t):
    sum_ = 0.;
    omega = h*.5
    while omega<1:
        l = (-Tx*np.log(omega))**(1/k)
        factor = np.exp(-(l**n)/(Tl))*Tx/(k*l**(k-1))*omega*np.exp(-omega*t)
        sum_+=factor*h
        omega+=h
    return sum_

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    t_0 = time.time()

    idx = int(args.Idx)

    Tx = .1
    k,n,Tl = np.array(np.loadtxt('iteration_indices_polynomial.txt')[idx])
    k = int(k)
    n = int(n)
    print(k,n,Tl,flush=True)
    h = 1e-9
    t0 = 1e1
    tf = 1e9
    trange = np.logspace(np.log10(t0),np.log10(tf),30)
    integral = np.zeros((len(trange)))
    for kt,t in enumerate(trange):
        integral[kt] = integrate(h,Tl,Tx,k,n,t)
        print(t,integral[kt],flush=True)


    f = h5py.File('~/Repositories/fluctuating_potential_repo+data/data/asymptotics/asymptotic_integral/integral_{}.h5'.format(idx),'w')
    x_ = f.create_dataset('integral',integral.shape)
    x_[...] = integral
    f.close()

    f = h5py.File('~/Repositories/fluctuating_potential_repo+data/asymptotics/asymptotic_integral/metadata.h5'.format(idx),'w')
    Tx_ = f.create_dataset('Tx',(1,))
    Tx_[...] = Tx
    h_ = f.create_dataset('h',(1,))
    h_[...] = h
    t0_ = f.create_dataset('t0',(1,))
    t0_[...] = t0
    tf_ = f.create_dataset('tf',(1,))
    tf_[...] = tf
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes'.format((t_f-t_0)/60),flush=True)

if __name__ == "__main__":
    main(sys.argv)
