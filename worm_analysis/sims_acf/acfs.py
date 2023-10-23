import numpy as np
import numpy.ma as ma
import argparse
import sys
#change path_to_utils and path_to_data accordingly
sys.path.append("path_to_utils/")
import stats
import h5py

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm',help='index',default=0,type=int)
    args=parser.parse_args()
    kw = int(args.worm)
    print(kw,flush=True)
    print('Load data',flush=True)

    frameRate=16.
    data_dt=1/frameRate

    ctraj_path = 'path_to_data/worm_analysis/ctrajs_1000_clusters/'
    f = h5py.File(ctraj_path+'/c_traj_w.h5','r')
    mD = f['MetaData']
    n_clusters = np.array(mD['n_clusters'],dtype=int)[0]
    delay = np.array(mD['delay'],dtype=int)[0]
    ctraj_w = ma.array(f['ctraj_w'])
    ctraj_w_mask = ma.array(f['ctraj_w_mask'])
    f.close()
    ctraj_w[ctraj_w_mask==1]=ma.masked

    lags = np.unique(np.array(np.logspace(0,np.log10(len(ctraj_w[0])),3000),dtype=int))
#     lags = np.arange(len(ctraj_w[0]))
    C_data = stats.acf(ctraj_w[kw],lags)

    f = h5py.File('path_to_data/worm_analysis/sims_fpts/sims_w_{}.h5'.format(kw),'r')
    sims = np.array(f['sims'])
    sims_tv = np.array(f['sims_tv'])
    f.close()

    n_sims = 1000
    C_sims = np.zeros((n_sims,len(lags)))
    C_sims_tv = np.zeros((n_sims,len(lags)))
    for ks in range(n_sims):
        C_sims[ks] = stats.acf(sims[ks],lags)
        C_sims_tv[ks] = stats.acf(sims_tv[ks],lags)
        if ks%10==0:
            print(ks,flush=True)

    print('Saving results',flush=True)
    f = h5py.File('path_to_data/worm_analysis/sims_acfs/acfs_w_{}.h5'.format(kw),'w')
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
