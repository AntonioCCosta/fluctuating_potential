import numpy as np

cluster_range = np.logspace(1,3.5,11)
K_range = np.arange(1,9)
all_indices=[]
for K in K_range:
    for N in cluster_range:
        all_indices.append([K,N])
all_indices = np.array(np.vstack(all_indices),dtype=float)
np.savetxt('iteration_indices.txt',all_indices)
print(len(all_indices))
