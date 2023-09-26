import numpy as np
import argparse
import sys
import time
import h5py
import numba


def main(argv):
    #converting .out files into .h5 files
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    parser.add_argument('-max_lines','--Max',help='max_n_lines',default=1000000,type=int)

    args=parser.parse_args()
    idx = int(args.Idx)

    #change output_path
    f = h5py.File('output_path/metadata.h5','r')
    dt = np.array(f['dt'])[0]
    f.close()

    max_n_lines = int(args.Max)

    print_stride = int(max_n_lines)/100

    fpts_pair=[]
    with open("output_path/fptds_{}.out".format(idx)) as f:
        count=0
        while count<max_n_lines:
            count += 1
            line = f.readline()
            t_s = line.split('\n')[0]
            if len(t_s)>0:
                t_s_split = t_s.split(';')[:-1]
                for tss in t_s_split:
                    tss_split = tss.split('|')
                    if len(tss_split)>0:
                        fpt = tss_split[0]
                        s = tss_split[1]
                        fpts_pair.append([fpt,s])
            if count%print_stride==0:
                print('Got {:.2f} for the results.'.format(count/max_n_lines),flush=True)
            if not line:
                print('End of the file',flush=True)
                break

    fpts_pair = np.array(np.vstack(fpts_pair),dtype=float)
    fpts = fpts_pair[:,0]*dt
    s_sample = fpts_pair[:,1]

    f = h5py.File('outut_path/combined_results_{}.h5'.format(idx),'w')
    s_ = f.create_dataset('s_sample',s_sample.shape)
    s_[...] = s_sample
    fpts_ = f.create_dataset('fpts',fpts.shape)
    fpts_[...] = fpts
    f.close()
    print('Saved {} results.'.format(max_n_lines))

if __name__ == "__main__":
    main(sys.argv)
