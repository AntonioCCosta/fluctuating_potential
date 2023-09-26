import numpy as np

k_range = np.arange(2,5)
n_range = np.arange(2,5)
Tl_range = np.arange(.025,0.21,.025)
all_indices=[]
for k in k_range:
    for n in n_range:
        for Tl in Tl_range:
            all_indices.append([k,n,Tl])
all_indices = np.array(np.vstack(all_indices),dtype=float)
np.savetxt('iteration_indices_polynomial.txt',all_indices)
print(len(all_indices))
