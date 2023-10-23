import numpy as np
import numpy.ma as ma
import argparse
import sys
#change path_to_utils and path_to_data accordingly
sys.path.append("path_to_utils/")
import stats
import h5py
import pickle

def transform_to_ctraj(phi2_sim_traj,phi2_thresh,ctraj_dict_pos,ctraj_dict_neg):
    sim_ctraj = np.zeros(phi2_sim_traj.shape)
    for kt in range(len(phi2_sim_traj)):
        val = phi2_sim_traj[kt]
        if val>=phi2_thresh:
            idx = np.argmin(np.abs(list(ctraj_dict_pos.keys())-val))
            new_val = ctraj_dict_pos[list(ctraj_dict_pos.keys())[idx]]
        if val<phi2_thresh:
            idx = np.argmin(np.abs(list(ctraj_dict_neg.keys())-val))
            new_val = ctraj_dict_neg[list(ctraj_dict_neg.keys())[idx]]
        sim_ctraj[kt] = new_val
    return sim_ctraj

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

    f = open('ctraj_path/ctraj_dict_neg.pkl', 'rb')
    ctraj_dict_neg = pickle.load(f)
    f.close()
    f = open('ctraj_path/ctraj_dict_pos.pkl', 'rb')
    ctraj_dict_pos = pickle.load(f)
    f.close()


    f = h5py.File('ctraj_path/c_traj_w.h5','r')
    phi2_thresh = np.array(f['MetaData/phi2_thresh'])
    phi2_traj = np.array(f['phi2_traj'])
    phi2 = np.array(f['eigfunctions'])[:,1]
    f.close()


    f = h5py.File('path_to_data/worm_data/symbol_sequence_simulations.h5','r')
    sims = np.array(f[str(kw)]['sims'],dtype=int)
    f.close()
    lags_sim = np.arange(1,len(sims[0]))
    n_sims=len(sims)
    C_sims= np.zeros((n_sims,lags_sim.shape[0]))
    for ksim in range(n_sims):
        phi2_sim_traj = phi2[sims[ksim]]
        sim_ctraj = transform_to_ctraj(phi2_sim_traj,phi2_thresh,ctraj_dict_pos,ctraj_dict_neg)
        C_sims[ksim] = stats.acf(sim_ctraj,lags_sim)
        if ksim%50==0:
            print(ksim)

    print('Saving results',flush=True)
    f = h5py.File('path_to_data/worm_analysis/sims_acfs/acfs_Markov_w_{}.h5'.format(kw),'w')
    Cs_ = f.create_dataset('C_sims',C_sims.shape)
    Cs_[...] = C_sims
    l_ = f.create_dataset('lags_sim',lags_sim.shape)
    l_[...] = lags_sim
    f.close()


if __name__ == "__main__":
    main(sys.argv)
