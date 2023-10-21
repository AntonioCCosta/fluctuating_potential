import numpy as np
import argparse
import sys
import time
import h5py

#change path_to_data to the data path

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)
    f = h5py.File('path_to_data/Poisson/sims_fpts_be/dts_{}.h5'.format(idx),'r')
    fpts = np.array(f['dts_sim'])
    f.close()
    print(fpts.shape,fpts.max(),flush=True)

    bins=np.logspace(-2,7,200)
    freqs,bin_edges = np.histogram(fpts,bins)
    centers_t = (bin_edges[:-1]+bin_edges[1:])/2

    f = h5py.File('path_to_data/Poisson/sims_fpts_be/fptd_{}.h5'.format(idx),'w')
    f_ = f.create_dataset('freqs',freqs.shape)
    f_[...] = freqs
    ct_ = f.create_dataset('centers_t',centers_t.shape)
    ct_[...] = centers_t
    b_ = f.create_dataset('bins',bins.shape)
    b_[...] = bins
    f.close()


if __name__ == "__main__":
    main(sys.argv)
