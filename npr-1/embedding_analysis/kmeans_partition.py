import numpy as np
import numpy.ma as ma
import sys
sys.path.append('/home/a/antonio-costa/TransferOperators/bridging_scales_manuscript/utils/')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import h5py
import argparse
from scipy.spatial import distance as sdist
import time

def main(argv):
    start_t = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    args=parser.parse_args()
    idx = args.Idx

    K,idx = np.array(np.loadtxt('/home/a/antonio-costa/BehaviorModel/npr-1/embedding_analysis/iteration_indices_K_1_20.txt')[idx],dtype=int)
    print(K,idx)

    seed_range = np.array(np.logspace(1,3.5,11),dtype=int)
    print('Loading data ...',flush=True)
    
    f = h5py.File('/bucket/StephensU/antonio/npr-1_data/resampled_results.h5','r')
    print(list(f.keys()))
    frameRate = np.array(f['new_frameRate'])[0]
    dt=1./frameRate
    worm_labels = list(f.keys())[1:]
    tseries_w=[]
    for worm in worm_labels:
        ts = ma.masked_invalid(np.array(f[worm]))
        tseries_w.append(ts)
    f.close()


    tseries = tseries_w[idx]
    tseries[tseries==0]=ma.masked


        
    print('Compute entropies ...')
    H_K_s = np.zeros(len(seed_range))
    prob_K_s = np.zeros(len(seed_range))
    h_K_s = np.zeros(len(seed_range))
    Ipred_K_s = np.zeros(len(seed_range))
    eps_K_s = np.zeros(len(seed_range))
    traj_matrix = embed.trajectory_matrix(tseries,K=K-1)
    print(traj_matrix.shape)
    for ks,n_seeds in enumerate(seed_range):
        labels,centers = cl.kmeans_knn_partition(traj_matrix,n_seeds,return_centers=True)
        labels = ma.array(labels,dtype=int)
        max_epsilon = np.mean([np.max(np.linalg.norm(traj_matrix[labels==kc]-centers[kc],axis=0)) for kc in np.sort(np.unique(labels.compressed()))])
        
        P = op_calc.transition_matrix(labels,1)
        prob = op_calc.stationary_distribution(P)
        H = -np.sum(prob*np.log(prob))
        h = op_calc.get_entropy(labels)
        prob_K_s[ks] = np.mean(prob)
        H_K_s[ks] = H
        h_K_s[ks] = h
        Ipred_K_s[ks] = H-h
        eps_K_s[ks] = max_epsilon
        print(n_seeds,flush=True)
    
    print('Saving results ...')
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/npr-1/embedding_results/entropic_properties_K_{}_{}.h5'.format(K,idx),'w')
    probs_ = f.create_dataset('probs',prob_K_s.shape)
    probs_[...] = prob_K_s
    H_ = f.create_dataset('entropies',H_K_s.shape)
    H_[...] = H_K_s
    h_ = f.create_dataset('entropy_rates',h_K_s.shape)
    h_[...] = h_K_s
    eps_ = f.create_dataset('eps_scale',eps_K_s.shape)
    eps_[...] = eps_K_s
    seeds_ = f.create_dataset('seed_range',seed_range.shape)
    seeds_[...] = seed_range
    Ipreds_ = f.create_dataset('Ipreds',Ipred_K_s.shape)
    Ipreds_[...] = Ipred_K_s
    f.close()
    end_t = time.time()
    print('It took {} s.'.format(end_t-start_t))

if __name__ == "__main__":
    main(sys.argv)
