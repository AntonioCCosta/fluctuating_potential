import numpy as np
import argparse
import sys
import time
import h5py

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-Ts_idx','--Ts_Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    Ts_idx = int(args.Ts_Idx)

    params = np.loadtxt('./iteration_Ts.txt')
    Ts_range = np.unique(params[:,0])
    indices = np.array(np.unique(params[:,1]),dtype=int)


    dts_Ts = []
    idx=0
    for Ts in Ts_range:
        dts=[]
        for k in indices:
            #change output_path
            f = h5py.File('output_path/combined_results_{}.h5'.format(idx),'r')
            fpts = np.array(f['fpts'])
            f.close()
            dts.append(fpts)
            idx+=1
        dts_Ts.append(np.hstack(dts))

    fpts = dts_Ts[Ts_idx]

    print(fpts.shape,fpts.max(),flush=True)

    bins=np.logspace(-2,7,200)
    #bins=np.logspace(-2,7,100)
    freqs,bin_edges = np.histogram(fpts,bins)
    centers_t = (bin_edges[:-1]+bin_edges[1:])/2

    print(bins.shape,flush=True)
    #change output_path
    f = h5py.File('output_path/fptd_Ts_idx_{}.h5'.format(Ts_idx),'w')
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
