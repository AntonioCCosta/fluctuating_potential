import numpy as np
import argparse
import sys
import time
import h5py
import numba
from numba import jit,prange,objmode
@jit(nopython=True,parallel=True)
def simulate_DW(x0s,s0s,dt,tau_s,Tx,Ts,mu_s,max_iters): #results are printed directly on an output file in the cluster
    n_sims = len(x0s)
    for ks in prange(n_sims):
        x0 = x0s[ks]
        s0 = s0s[ks]
        x=x0
        s=s0
        if x0<0:
            x_flag_left=True
            x_flag_right=False
            t0_left=0
        if x0>0:
            x_flag_right=True
            x_flag_left=False
            t0_right=0
        for i in range(max_iters):
            new_x = x - 4*s**2*x*(x**2-1)*dt + np.sqrt(2*Tx)*np.random.normal(0, np.sqrt(dt))
            new_s = s - (1/tau_s)*(s-mu_s)*dt + np.sqrt(2*Ts*(1/tau_s))*np.random.normal(0, np.sqrt(dt))
            if new_x>0 and x_flag_left:
                with objmode():
                    output_string = str(i-t0_left)+"|"+str(new_s)+";"
                    print(output_string)
                x_flag_left=False
            if new_x<-1 and x_flag_left == False:
                x_flag_left=True
                t0_left = i
            if new_x<0 and x_flag_right:
                with objmode():
                    output_string = str(i-t0_right)+"|"+str(new_s)+";"
                    print(output_string)
                x_flag_right=False
            if new_x>1 and x_flag_right == False:
                x_flag_right=True
                t0_right = i
            x = new_x
            s = new_s


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)
    Ts,k_idx = np.loadtxt('iteration_Ts.txt')[idx]
    n_sims=1000
    Tx=1e-3
    mu_s = np.sqrt(Tx)
    s0s = np.random.normal(mu_s,np.sqrt(Ts),n_sims)
    x0s = np.random.choice([-1,1],n_sims)
    dt=1e-3
    eps_s = 1000
    maxT = int(1e7)
    tau_s = maxT*eps_s
    max_iters = int(maxT/dt)

    if idx==0:
        #change output_path
        f = h5py.File('output_path/metadata.h5','w')
        Tx_ = f.create_dataset('Tx',(1,))
        Tx_[...] = Tx
        dt_ = f.create_dataset('dt',(1,))
        dt_[...] = dt
        tau_s_ = f.create_dataset('tau_s',(1,))
        tau_s_[...] = tau_s
        maxT_ = f.create_dataset('maxT',(1,))
        maxT_[...] = maxT
        ms_ = f.create_dataset('mu_s',(1,))
        ms_[...] = mu_s
        f.close()

    fpts = simulate_DW(x0s,s0s,dt,tau_s,Tx,Ts,mu_s,max_iters)

if __name__ == "__main__":
    main(sys.argv)
