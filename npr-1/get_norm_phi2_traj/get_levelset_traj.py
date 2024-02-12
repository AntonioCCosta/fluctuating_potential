#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import sys
import os
sys.path.append('/home/a/antonio-costa/BehaviorModel/utils/')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import worm_dynamics as worm_dyn
import stats
import h5py

np.random.seed(0)


f = h5py.File('/bucket/StephensU/antonio/npr-1_data/resampled_results.h5','r')
frameRate = np.array(f['new_frameRate'])[0]
dt= 1./frameRate
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
lcs,P = op_calc.transition_matrix(labels_traj,delay,return_connected=True)
final_labels = op_calc.get_connected_labels(labels_traj,lcs)
n_modes=10
inv_measure = op_calc.stationary_distribution(P)
R = op_calc.get_reversible_transition_matrix(P)
eigvals,eigvecs = op_calc.sorted_spectrum(R,k=n_modes)
sorted_indices = np.argsort(eigvals.real)[::-1]
eigvals = eigvals[sorted_indices][1:].real
eigvals[np.abs(eigvals-1)<1e-12] = np.nan
eigvals[eigvals<1e-12] = np.nan
eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)
eigfunctions_traj = ma.array(eigfunctions)[final_labels,:]
eigfunctions_traj[final_labels.mask] = ma.masked
phi2 = eigfunctions[:,1]

thresh_range,rho_c,c_thresh,kmeans_labels = op_calc.optimal_partition(eigfunctions[:,1],inv_measure,P,return_rho=True)

phi2_traj = eigfunctions_traj[:,1]


sel = eigfunctions[:,1]<=c_thresh
c_negative = np.linspace(-2,0,sel.sum())
d_neg = {}
for k in range(c_negative.shape[0]):
    d_neg[np.sort(eigfunctions[sel,1])[k]] = c_negative[k]
    
sel = eigfunctions[:,1]>=c_thresh
c_positive = np.linspace(0,2,sel.sum())
d_pos = {}
for k in range(c_positive.shape[0]):
    d_pos[np.sort(eigfunctions[sel,1])[k]] = c_positive[k]
    
ctraj = ma.zeros(phi2_traj.shape[0])
sel = np.logical_and(phi2_traj<c_thresh,~phi2_traj.mask)
ctraj[sel] = np.array([d_neg[kc] for kc in phi2_traj[sel]])
sel = np.logical_and(phi2_traj>=c_thresh,~phi2_traj.mask)
ctraj[sel] = np.array([d_pos[kc] for kc in phi2_traj[sel]])
sel_mask = phi2_traj.mask
ctraj[sel_mask] = ma.masked

output_path = '/flash/StephensU/antonio/BehaviorModel/npr-1/ctrajs_562_clusters_delay_{}/'.format(delay)

f = h5py.File(output_path+'/c_traj.h5','w')
mD = f.create_group('MetaData')
N_ = mD.create_dataset('n_clusters',(1,))
N_[...] = n_clusters
d_ = mD.create_dataset('delay',(1,))
d_[...] = delay
ts_ = mD.create_dataset('phi2_thresh',(1,))
ts_[...] = c_thresh
tsc_ = mD.create_dataset('ctraj_thresh',(1,))
tsc_[...] = 0
kmeans_labels_ = f.create_dataset('kmeans_labels',kmeans_labels.shape)
kmeans_labels_[...] = kmeans_labels

final_labels_ = f.create_dataset('final_labels',final_labels.shape)
final_labels_[...] = final_labels
phi2traj_ = f.create_dataset('phi2_traj',phi2_traj.shape)
phi2traj_[...] = phi2_traj
mask_ = f.create_dataset('mask',phi2_traj.shape)
mask_[...] = phi2_traj.mask
ctraj_mask = np.zeros(ctraj.shape)
ctraj_mask[ctraj.mask]=1
ctraj_ = f.create_dataset('ctraj',ctraj.shape)
ctraj_[...] = ctraj
ctrajm_ = f.create_dataset('ctraj_mask',ctraj_mask.shape)
ctrajm_[...] = ctraj_mask
eigfun_ = f.create_dataset('eigfunctions',eigfunctions.shape)
eigfun_[...] = eigfunctions
f.close()


import pickle
with open(output_path+'ctraj_dict_neg.pkl', 'wb') as f:
    pickle.dump(d_neg,f)
with open(output_path+'ctraj_dict_pos.pkl', 'wb') as f:
    pickle.dump(d_pos,f)
