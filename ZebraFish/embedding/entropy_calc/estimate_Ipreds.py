import numpy as np
import numpy.ma as ma
import h5py
import sys
import argparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/a/antonio-costa/BehaviorModel/utils/')
import delay_embedding as embed
import operator_calculations as op_calc
import clustering_methods as cl
import stats


from scipy.sparse import csc_matrix 
from scipy.sparse import diags,identity,coo_matrix
def get_entropy_P(P):
    #get dtrajs to deal with possible nans
    P = csc_matrix(P)
    probs = op_calc.stationary_distribution(P)
    logP = P.copy()
    logP.data = np.log(logP.data)
    return (-diags(probs).dot(P.multiply(logP))).sum()

def get_h_Ipred(labels):
    P = op_calc.transition_matrix(labels,1)
    pi = op_calc.stationary_distribution(P)
    sel = pi>0
    H = np.sum(-pi[sel]*np.log(pi[sel]))
    h = get_entropy_P(P)
    return h,H-h


def get_compressed_labels(labels_cond):
    labels_comp = []
    for labels in labels_cond:
        bf = np.arange(len(labels))[~labels.mask][-1]
        labels_here = ma.copy(labels[:bf+1])
        if ~labels_here.mask[0]:
            labels_here[0] = ma.masked
        labels_comp.append(labels_here)
    return ma.hstack(labels_comp)

def sample_each_cond(labels_fish,labels_shuffle_fish,condition_recs,n_bouts,n_shuffle):
    ks = np.random.randint(n_shuffle)
    labels_all = []
    labels_all_shuffle = []
    for kc in range(len(condition_recs)):
        c0,cf = condition_recs[kc]
        labels_cond = labels_fish[ks,c0:cf,:]
        labels_comp = get_compressed_labels(labels_cond)
        labels_cond_shuffle = labels_shuffle_fish[ks,c0:cf,:]
        labels_comp_shuffle = get_compressed_labels(labels_cond_shuffle)
        b0 = np.random.randint(0,len(labels_comp)-n_bouts)
        labels_sample = labels_comp[b0:b0+n_bouts]
        labels_sample_shuffle = labels_comp_shuffle[b0:b0+n_bouts]
        labels_all.append(labels_sample)
        labels_all_shuffle.append(labels_sample_shuffle)
    labels_all = ma.hstack(labels_all)
    labels_all_shuffle = ma.hstack(labels_all_shuffle)
    return labels_all,labels_all_shuffle


from numba import jit
@jit(nopython=True)
def tm_seg(X,K):
    '''
    Build a trajectory matrix
    X: N x dim data
    K: the number of delays
    out: (N-K)x(dim*K) dimensional
    '''
    tm=np.zeros(((len(X)-K-1),X.shape[1]*(K+1)))
    for t in range(len(X)-K-1):
        x = X[t:t+K+1,:][::-1]
        x_flat = x.flatten()
        tm[t] = x_flat
    return tm

def get_traj_matrix(bouts,K):
    traj_matrix= ma.zeros((len(bouts),bouts.shape[1]*(K+1)))
    traj_matrix[int(np.floor((K)/2)):-int(np.ceil((K)/2)+1)] = tm_seg(bouts,K=K)
    traj_matrix[:int(np.floor((K)/2))] = ma.masked
    traj_matrix[-int(np.ceil((K)/2)+1):] = ma.masked
    traj_matrix[traj_matrix==0] = ma.masked
    return traj_matrix

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    
    idx = args.Idx
    
    K,N = np.array(np.loadtxt('iteration_indices.txt')[idx],dtype=int)

    n_pca_modes=20
    f= h5py.File('/bucket/StephensU/antonio/BehaviorModel/Zebrafish/zebrafish_data.h5','r')
    pca_fish = ma.masked_invalid(np.array(f['pca_fish']))[:,:,:n_pca_modes]
    f.close()
    pca_fish[pca_fish==0]=ma.masked
    
    print(pca_fish.shape)
    
    traj_matrix_all = get_traj_matrix(ma.vstack(pca_fish),K=K-1)
    labels_all = cl.kmeans_knn_partition(traj_matrix_all,n_seeds=N)
    
    labels_fish = labels_all.reshape(pca_fish.shape[0],pca_fish.shape[1])
    
    print(traj_matrix_all.shape,labels_fish.shape)
    
    n_fish = len(pca_fish)
    h_fish = np.zeros(n_fish)
    Ipred_fish = np.zeros(n_fish)
    for kf in range(n_fish):
        labels = labels_fish[kf]
        h,Ipred = get_h_Ipred(labels)
        h_fish[kf] = h
        Ipred_fish[kf] = Ipred
        print(kf)
    
    
    print('Saving results',flush=True)
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/Zebrafish/Ipreds/Ipreds_{}.h5'.format(idx),'w')
    I_ = f.create_dataset('Ipreds',Ipred_fish.shape)
    I_[...] = Ipred_fish
    h_ = f.create_dataset('hs',h_fish.shape)
    h_[...] = h_fish
    f.close()
       

if __name__ == "__main__":
    main(sys.argv)
