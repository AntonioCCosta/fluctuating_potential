import numpy as np

Tx=1e-3
Ts_range = np.linspace(Tx/4,Tx*2,10)
indices = np.arange(10)
l=[]
for Ts in Ts_range:
    for idx in indices:
        l.append([Ts,idx])


np.savetxt('iteration_Ts.txt',np.vstack(l))
print(len(l))
