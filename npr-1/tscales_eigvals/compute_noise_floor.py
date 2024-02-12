import numpy as np
import numpy.ma as ma
import argparse
import sys
import time
import h5py
sys.path.append('/home/a/antonio-costa/TransferOperators/bridging_scales_manuscript/utils/')
import operator_calculations as op_calc


def shuffle_masked(labels):
    segs = op_calc.segment_maskedArray(labels)
    labels_shuffle = ma.zeros(labels.shape,dtype=int)
    labels_shuffle[labels.mask] = ma.masked
    for seg in segs:
        t0,tf=seg
        indices = np.random.randint(t0,tf,tf-t0)
        labels_shuffle[t0:tf] = labels[indices]
    return labels_shuffle


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm',help='worm',default=0,type=int)
    args=parser.parse_args()
    
    kw = int(args.worm)
    
    f = h5py.File('/bucket/StephensU/antonio/npr-1_data/resampled_results.h5','r')
    frameRate = np.array(f['new_frameRate'])[0]
    worm_labels = list(f.keys())[1:]
    n_worms = len(worm_labels)
    tseries_w=[]
    for worm in worm_labels:
        ts = ma.masked_invalid(np.array(f[worm]))
        tseries_w.append(ts)
    f.close()
    
    
    n_clusters=562
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/npr-1/labels_K_8_N_562.h5','r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    f.close()

    labels_traj[mask_traj] = ma.masked

    labels_w=[]
    t0=0
    for w_idx in range(n_worms):
        labels_here = labels_traj[t0:t0+len(tseries_w[w_idx])].copy()
        t0+=len(tseries_w[w_idx])
#         print(t0,labels_here[:100],flush=True)
        labels_w.append(labels_here)
        
    labels = labels_w[kw]
    
    print(labels[:50],flush=True)
    
    n_modes = 50
    dt = 1./frameRate
    nstates = n_clusters
    
    delay_range = np.unique(np.array(np.logspace(0,3,100),dtype=int))
    print(delay_range.shape,flush=True)
    
    n_shuffle = 100
    
    ts_traj_shuffle = np.zeros((len(delay_range),n_shuffle))
    eigvals_traj_shuffle = np.zeros((len(delay_range),n_shuffle))
    ts_traj_w = np.zeros((len(delay_range),n_modes))
    eigvals_traj_w = np.zeros((len(delay_range),n_modes))
    
    for kd,delay in enumerate(delay_range):
        P = op_calc.transition_matrix(labels,delay)
        R = op_calc.get_reversible_transition_matrix(P)
        eigvals,eigvecs = op_calc.sorted_spectrum(R,k=n_modes+1)
        timp = -(delay*dt)/np.log(eigvals[1:].real)
        ts_traj_w[kd,:] = timp
        eigvals_traj_w[kd,:] = eigvals[1:].real
        for ks in range(n_shuffle):
            labels_shuffle = shuffle_masked(labels)
            P = op_calc.transition_matrix(labels_shuffle,delay)
            R = op_calc.get_reversible_transition_matrix(P)
            eigvals,eigvecs = op_calc.sorted_spectrum(R,k=2)
            timp = -(delay*dt)/np.log(eigvals[1].real)
            ts_traj_shuffle[kd,ks] = timp
            eigvals_traj_shuffle[kd,ks] = eigvals[1].real        
        print(kd,delay,flush=True)
        
    print('Saving results',flush=True)
    f = h5py.File('/flash/StephensU/antonio/npr-1/tscales_noise_floor/results_{}.h5'.format(kw),'w')
    ts_s = f.create_dataset('ts_traj_shuffle',ts_traj_shuffle.shape)
    ts_s[...] = ts_traj_shuffle
    eigs_s = f.create_dataset('eigvals_traj_shuffle',eigvals_traj_shuffle.shape)
    eigs_s[...] = eigvals_traj_shuffle
    ts_w = f.create_dataset('ts_traj_w',ts_traj_w.shape)
    ts_w[...] = ts_traj_w
    eigs_w = f.create_dataset('eigvals_traj_w',eigvals_traj_w.shape)
    eigs_w[...] = eigvals_traj_w
    ds = f.create_dataset('delay_range',delay_range.shape)
    ds[...] = delay_range
    n = f.create_dataset('n_clusters',(1,))
    n[...] = n_clusters
    f.close()
    
if __name__ == "__main__":
    main(sys.argv)
