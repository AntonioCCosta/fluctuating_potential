import numpy as np
import numpy.ma as ma
import h5py
import sys
import argparse
from scipy.integrate import odeint
sys.path.append('/home/a/antonio-costa/TransferOperators/bridging_scales_manuscript/utils/')
import operator_calculations as op_calc
import delay_embedding as embed
import clustering_methods as cl
from sklearn.decomposition import FastICA

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_seeds','--N',help="Number of states",default=1000,type=int)
    args=parser.parse_args()
    n_clusters = args.N
    
    f = h5py.File('/bucket/StephensU/antonio/npr-1_data/resampled_results.h5','r')
    print(list(f.keys()))
    frameRate = np.array(f['new_frameRate'])[0]
    worm_labels = list(f.keys())[1:]
    tseries_w=[]
    for worm in worm_labels:
        ts = ma.masked_invalid(np.array(f[worm]))
        tseries_w.append(ts)
    f.close()


    masked_ts_w = []
    for worm in np.arange(len(tseries_w)):
        ts_w = tseries_w[worm]
        ts_w[0] = ma.masked
        ts_w[-1] = ma.masked
        masked_ts_w.append(ts_w)

    tseries_all = ma.vstack(masked_ts_w)
    
    K_star=8
    
    m_star=9

    #tseries_all = tseries_all[:1000]
    
    traj_matrix = embed.trajectory_matrix(tseries_all,K=K_star)

    sel = ~np.any(traj_matrix.mask,axis=1)
    X = traj_matrix[sel]
    transformer = FastICA(n_components=m_star,random_state=0,max_iter = 10000,tol = 1e-10)
    X_transformed = transformer.fit_transform(X)
    
    components = transformer.components_
    phspace = ma.zeros((traj_matrix.shape[0],m_star))
    phspace[sel] = X_transformed
    phspace[~sel] = ma.masked
        
    print('Saving embedding results...')

    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/npr-1/phspace_K_{}_m_{}.h5'.format(K_star,m_star),'w')
    modes_ = f.create_dataset('modes',components.shape)
    modes_[...] = components
    ph_ = f.create_dataset('phspace',phspace.shape)
    ph_[...] = phspace
    traj_ = f.create_dataset('traj_matrix',traj_matrix.shape)
    traj_[...] = traj_matrix
    f.close()

    print('Partitioning...')
    
    #cluster_range = np.array(np.arange(500,3100,500),dtype=int)
    cluster_range=[562,]
    for n_clusters in cluster_range:
        labels_traj,centers_traj = cl.kmeans_knn_partition(traj_matrix,n_seeds=n_clusters,return_centers=True)
        labels_phspace,centers_phspace = cl.kmeans_knn_partition(phspace,n_seeds=n_clusters,return_centers=True)
        f = h5py.File('/flash/StephensU/antonio/BehaviorModel/npr-1/labels_K_{}_N_{}.h5'.format(K_star,n_clusters),'w')
        ltraj_ = f.create_dataset('labels_traj',labels_traj.shape)
        ltraj_[...] = labels_traj
        mtraj_ = f.create_dataset('mask_traj',labels_traj.shape,dtype=bool)
        mtraj_[...] = labels_traj.mask
        ctraj_ = f.create_dataset('centers_traj',centers_traj.shape)
        ctraj_[...] = centers_traj
        lp_ = f.create_dataset('labels_phspace',labels_phspace.shape)
        lp_[...] = labels_phspace
        mp_ = f.create_dataset('mask_phspace',labels_phspace.shape,dtype=bool)
        mp_[...] = labels_phspace.shape
        cp_ = f.create_dataset('centers_phspace',centers_phspace.shape)
        cp_[...] = centers_phspace
        f.close()

        
if __name__ == "__main__":
    main(sys.argv)
