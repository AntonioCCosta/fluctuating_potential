import numpy as np

Tx=1e-3
Ts = 0.0005

eps_range = np.logspace(-4,1,6)
indices = np.arange(10)
l=[]
for eps in eps_range:
    for idx in indices:
        l.append([Ts,eps,idx])


np.savetxt('iteration_eps.txt',np.vstack(l))
print(len(l))
