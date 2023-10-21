import numpy as np
import numpy.ma as ma
import argparse
import sys
import time
import h5py
#change path_to_utils and path_to_data accordingly
sys.path.append("path_to_utils/")
import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

def potential(x,s):
    return s**2*(x**2-1)**2

def exppot(x, Tx,s,sign=-1, fun=lambda z: 1):
    return np.exp(sign*potential(x,s)/Tx)*fun(x)

def omega_be(Tx,s):
    z = np.linspace(-20, 0, 1000)
    iarr = cumtrapz(exppot(z,Tx,s), z, initial=0)
    ifun = interp1d(z, iarr)

    y = np.linspace(-1, 0, 1000)
    oarr = cumtrapz(exppot(y,Tx,s, sign=1, fun=ifun), y, initial=0)/Tx
    ofun = interp1d(y, oarr)

    return 1/(2*ofun(0))

acf = lambda t, s, Tx : np.exp(-2*omega_be(Tx,s)*t)
F = lambda s, t, Tx, Ts, mu_s : np.exp(-((s - mu_s)**2/(2*Ts)))*acf(t, s, Tx)

def C_corr(r_range,Cr,N):
    sum1 = (2*(N-np.arange(1,N-1))*Cr[1:N-1]).sum()
    Cr_range = np.zeros(len(r_range))
    for kr,r in enumerate(r_range):
        term1 = (1/N)*(1/N - 2/(N-kr))*(N*Cr[0]+sum1)
        if kr==0:
            sum2=0
        else:
            sum2 =(2*(kr-np.arange(1,kr-1))*Cr[1:kr-1]).sum()
        sum3 = (Cr[1:N-1]*(np.min([np.arange(1,N-1)+kr,N*np.ones(N-2)],axis=0)-np.max([kr*np.ones(N-2),np.arange(1,N-1)],axis=0))).sum()
        term2 = (2/(N*(N-kr)))*(kr*Cr[0]+sum2+sum3)
        Cr_range[kr] = Cr[kr]+term1+term2
    return Cr_range

def get_corrected_acf(Tx,mu_s,Ts,lags_corr,trange):
    srange = np.linspace(mu_s-20*Ts,mu_s+20*Ts,100)
    ds = srange[1]-srange[0]
    F_int= np.zeros(trange.shape)
    for s in srange:
        F_int+=F(s,trange,Tx,Ts,mu_s)*ds
    Cc = C_corr(lags_corr,F_int,len(lags_corr))
    return F_int,Cc


def bootstrap_norm_acfs(Cc,n_times=100):
    C_boot=[]
    for k in range(n_times):
        C_sample = Cc[np.random.randint(0,len(Cc),len(Cc))].mean(axis=0)
        C_norm = C_sample/C_sample[0]
        C_boot.append(C_norm)
    mean = Cc.mean(axis=0)
    mean = mean/mean[0]
    cil = np.percentile(C_boot,2.5,axis=0)
    ciu = np.percentile(C_boot,97.5,axis=0)
    return mean,cil,ciu

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)

    params = np.loadtxt('path_to_data/DW/sims_acfs_tau_s/iteration_Ts_eps.txt')
    Ts = params[idx,1]

    print(Ts,flush=True)

    f = h5py.File('path_to_data/DW/sims_acfs_tau_s/acfs_{}.h5'.format(idx),'r')
    C_nc_sims = ma.masked_invalid(np.array(f['C_nc_sims']))
    C_sims = ma.masked_invalid(np.array(f['C_sims']))
    lags = np.array(f['lags'],dtype=int)
    Tx = np.array(f['metaData/Tx'])[0]
    dt = np.array(f['metaData/dt'])[0]
    maxT = np.array(f['metaData/maxT'])[0]
    f.close()
    mu_s = np.sqrt(Tx)

    C_nc_sims = ma.compress_rows(C_nc_sims)
    C_sims = ma.compress_rows(C_sims)

    print(C_sims.shape,flush=True)

    mean_nc,cil_nc,ciu_nc = stats.bootstrap(C_nc_sims,n_times=1000)

    lags_corr = np.unique(np.array(np.arange(lags[0],lags[-1],1),dtype=int))[::100]

    trange = lags_corr*dt
    C_nc,C_corr = get_corrected_acf(Tx,mu_s,Ts,lags_corr,trange)

    mean_c,cil_c,ciu_c = bootstrap_norm_acfs(C_sims,n_times=1000)

    C_nc = np.vstack([mean_nc,cil_nc,ciu_nc]).T
    C_c = np.vstack([mean_c,cil_c,ciu_c]).T

    f = h5py.File('path_to_data/DW/sims_acfs_tau_s/bootstrapped_acfs_{}.h5'.format(idx),'w')
    cnc_ = f.create_dataset('C_nc_ci',C_nc.shape)
    cnc_[...] = C_nc
    cc_ = f.create_dataset('C_c_ci',C_c.shape)
    cc_[...] = C_c
    ccr_ = f.create_dataset('C_corr',C_corr.shape)
    ccr_[...] = C_corr
    tr_ = f.create_dataset('trange',trange.shape)
    tr_[...] = trange
    l_ = f.create_dataset('lags',lags.shape)
    l_[...] = lags
    f.close()

if __name__ == "__main__":
    main(sys.argv)
