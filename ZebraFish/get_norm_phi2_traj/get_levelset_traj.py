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

n_pca_modes=20
f= h5py.File('/bucket/StephensU/antonio/BehaviorModel/Zebrafish/zebrafish_data.h5','r')
print(f.keys())
cov = np.array(f['cov'])
data_means = np.array([f['data_means']])
eigvecs = np.array(f['eigvecs'])
max_shuffles = np.array(f['max_shuffs'])
pca_fish = ma.masked_invalid(np.array(f['pca_fish']))[:,:,:n_pca_modes]
var_exp = np.array(f['var_exp'])
f.close()
pca_fish[pca_fish==0]=ma.masked

K_star=5
traj_matrix_all = embed.trajectory_matrix(ma.vstack(pca_fish),K=K_star-1)

N=562
labels_all = cl.kmeans_knn_partition(traj_matrix_all,n_seeds=N)



delay = 3
print('Exctracting slow mode',flush=True)
lcs,P = op_calc.transition_matrix(labels_all,delay,return_connected=True)
inv_measure = op_calc.stationary_distribution(P)
final_labels = op_calc.get_connected_labels(labels_all,lcs)
n_modes=10
R = op_calc.get_reversible_transition_matrix(P)
eigvals,eigvecs = op_calc.sorted_spectrum(R,k=n_modes)
sorted_indices = np.argsort(eigvals.real)[::-1]
eigvals = eigvals[sorted_indices][1:].real
eigvals[np.abs(eigvals-1)<1e-12] = np.nan
eigvals[eigvals<1e-12] = np.nan
t_imp =  -(delay)/np.log(eigvals)
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

ctraj_fish = ctraj.reshape((pca_fish.shape[0],pca_fish.shape[1]))

ctraj_fish_mask = np.zeros(ctraj_fish.shape,dtype=int)
ctraj_fish_mask[ctraj_fish.mask] = 1

output_path = '/flash/StephensU/antonio/BehaviorModel/Zebrafish/ctrajs_{}_clusters_delay_{}/'.format(N,delay)

f = h5py.File(output_path+'/c_traj_fish.h5','w')
mD = f.create_group('MetaData')
N_ = mD.create_dataset('n_clusters',(1,))
N_[...] = N
d_ = mD.create_dataset('delay',(1,))
d_[...] = delay
ts_ = mD.create_dataset('phi2_thresh',(1,))
ts_[...] = c_thresh
tsc_ = mD.create_dataset('ctraj_thresh',(1,))
tsc_[...] = 0
ctrajs_ = f.create_dataset('ctraj_fish',ctraj_fish.shape)
ctrajs_[...] = ctraj_fish
ctrajs_mask_ = f.create_dataset('ctraj_fish_mask',ctraj_fish_mask.shape)
ctrajs_mask_[...] = ctraj_fish_mask
kmeans_labels_ = f.create_dataset('kmeans_labels',kmeans_labels.shape)
kmeans_labels_[...] = kmeans_labels
phi2traj_ = f.create_dataset('phi2_traj',phi2_traj.shape)
phi2traj_[...] = phi2_traj
mask_ = f.create_dataset('mask',phi2_traj.shape)
mask_[...] = phi2_traj.mask
eigfun_ = f.create_dataset('eigfunctions',eigfunctions.shape)
eigfun_[...] = eigfunctions
f.close()


import pickle
with open(output_path+'ctraj_dict_neg.pkl', 'wb') as f:
    pickle.dump(d_neg,f)
with open(output_path+'ctraj_dict_pos.pkl', 'wb') as f:
    pickle.dump(d_pos,f)
