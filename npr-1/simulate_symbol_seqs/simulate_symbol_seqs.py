import h5py
import numpy as np
import numpy.ma as ma
import os
import sys
sys.path.append('/home/a/antonio-costa/BehaviorModel/utils')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import worm_dynamics as worm_dyn
import stats


def simulate(P,state0,iters,lcs):
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
        state=new_state
        states[k]=state
    return lcs[states]

from joblib import Parallel, delayed

def simulate_parallel(P,state0,len_sim,lcs):
    return simulate(P,state0,len_sim,lcs)

f = h5py.File('/bucket/StephensU/antonio/npr-1_data/resampled_results.h5','r')
frameRate = np.array(f['new_frameRate'])[0]
data_dt= 1./frameRate
worm_labels = list(f.keys())[1:]
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

delay=10
ensemble_labels_w = []
t0=0
for kw in range(len(tseries_w)):
    ensemble_labels_w.append(labels_traj[t0:t0+len(tseries_w[kw])])
    t0+=len(tseries_w[kw])
    
    
n_sims = 1000
f = h5py.File('/flash/StephensU/antonio/npr-1/sims/symbol_sequence_simulations_1000_clusters.h5','w')

metaData = f.create_group('MetaData')
dl = metaData.create_dataset('delay',(1,))
dl[...] = delay
nc = metaData.create_dataset('n_clusters',(1,))
nc[...] = n_clusters

for kw,worm in enumerate(worm_labels):
    wg = f.create_group(str(worm))
    
    labels = ensemble_labels_w[kw]

    lcs,P = op_calc.transition_matrix(labels,delay,return_connected=True)

    final_labels = op_calc.get_connected_labels(labels,lcs)

    len_sim = int(len(labels)/delay)

    states0 = np.ones(n_sims,dtype=int)*final_labels.compressed()[0]

    sims = Parallel(n_jobs=50)(delayed(simulate_parallel)(P,state0,len_sim,lcs)
                               for state0 in states0)
    sims = np.array(sims)
    s_ = wg.create_dataset('sims',sims.shape)
    s_[...] = sims
    print(worm)
