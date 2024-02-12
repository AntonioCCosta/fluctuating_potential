import numpy as np
import numpy.ma as ma
import argparse
import sys
sys.path.append("/home/a/antonio-costa/BehaviorModel/utils/")
import stats
import h5py

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kf','--fish',help='index',default=0,type=int)
    args=parser.parse_args()
    kf = int(args.fish)
    print(kf,flush=True)
    print('Load data',flush=True)
    
    data_dt=1
    delay=3

    ctraj_path = '/flash/StephensU/antonio/BehaviorModel/Zebrafish/ctrajs_562_clusters_delay_{}/'.format(delay)
    f = h5py.File(ctraj_path+'/c_traj_fish.h5','r')
    mD = f['MetaData']
    n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
    delay = np.array(mD['delay'],dtype=int)[0]
    ctraj_fish = ma.array(f['ctraj_fish'])
    ctraj_fish_mask = ma.array(f['ctraj_fish_mask'])
    f.close()
    ctraj_fish[ctraj_fish_mask==1]=ma.masked
    
    lags = np.hstack(([0,],np.unique(np.array(np.logspace(0,4,200),dtype=int))))
    C_data = stats.acf(ctraj_fish[kf],lags)
    
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/Zebrafish/sims_fpts/sims_fish_{}.h5'.format(kf),'r')
    sims = np.array(f['sims'])
    sims_tv = np.array(f['sims_tv'])
    f.close()
    
    n_sims = 5000
    C_sims = np.zeros((n_sims,len(lags)))
    C_sims_tv = np.zeros((n_sims,len(lags)))
    for ks in range(n_sims):
        C_sims[ks] = stats.acf(sims[ks],lags)
        C_sims_tv[ks] = stats.acf(sims_tv[ks],lags)
        if ks%10==0:
            print(ks,flush=True)
        
    print('Saving results',flush=True)
    f = h5py.File('/flash/StephensU/antonio/BehaviorModel/Zebrafish/sims_acfs/acfs_fish_{}.h5'.format(kf),'w')
    Cd_ = f.create_dataset('C_data',C_data.shape)
    Cd_[...] = C_data
    Cs_tv_ = f.create_dataset('C_sims_tv',C_sims_tv.shape)
    Cs_tv_[...] = C_sims_tv
    Cs_ = f.create_dataset('C_sims',C_sims.shape)
    Cs_[...] = C_sims
    l_ = f.create_dataset('lags',lags.shape)
    l_[...] = lags
    f.close()
       

if __name__ == "__main__":
    main(sys.argv)
