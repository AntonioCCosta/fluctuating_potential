import numpy as np
import argparse
import sys
import time
import h5py

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-eps_idx','--Eps_Idx',help='eps_index',default=0,type=int)
    args=parser.parse_args()
    eps_idx = int(args.Eps_Idx)

    params = np.loadtxt('./iteration_eps.txt')
    Ts = np.unique(params[:,0])
    eps_range = np.unique(params[:,1])
    indices = np.array(np.unique(params[:,2]),dtype=int)

    dts_eps = []
    idx=0
    for eps in eps_range:
        dts=[]
        for k in indices:
            f = h5py.File('output_path/combined_results_{}.h5'.format(idx),'r')
            fpts = np.array(f['fpts'])
            f.close()
            dts.append(fpts)
            idx+=1
        dts_eps.append(np.hstack(dts))

    fpts = dts_eps[eps_idx]

    print(fpts.shape,fpts.max(),flush=True)

    bins=np.logspace(-2,7,200)
    #bins=np.logspace(-2,7,100)
    freqs,bin_edges = np.histogram(fpts,bins)
    centers_t = (bin_edges[:-1]+bin_edges[1:])/2

    print(bins.shape,flush=True)

    f = h5py.File('output_path/fptd_eps_idx_{}.h5'.format(eps_idx),'w')
    f_ = f.create_dataset('freqs',freqs.shape)
    f_[...] = freqs
    ct_ = f.create_dataset('centers_t',centers_t.shape)
    ct_[...] = centers_t
    b_ = f.create_dataset('bins',bins.shape)
    b_[...] = bins
    f.close()

    print('saved results',flush=True)


if __name__ == "__main__":
    main(sys.argv)
