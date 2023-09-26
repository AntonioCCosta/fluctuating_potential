import numpy as np

Tx=1e-2

Ts_range = 0.5*Tx/np.arange(0.05,0.45,0.05)
eps_range = eps_range = np.logspace(-2,2,9)
l=[]
for Ts in Ts_range:
    for eps in eps_range:
        l.append([Tx,Ts,eps])

np.savetxt('iteration_Ts_eps.txt',np.vstack(l))
print(len(l))
