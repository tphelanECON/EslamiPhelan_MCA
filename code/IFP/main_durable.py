"""
Durable consumption problem in three dimensions.
"""

import numpy as np
import pandas as pd
import time, classes

rho, r, eta, iota = -np.log(0.93), 0.06, 1/0.77-1, 0.01
theta, sigma = -np.log(0.95), np.sqrt(-2*np.log(0.95))*0.2
pbar, lambar = 1, 52
bnd, maxiter, tol, mono_tol = [[0,100],[-0.8,0.8],[0,12]], 5000, 10**-6, 10**-6

"""
Define number of runs and grid sizes. Want K/N[0] constant N as varies.
"""

runs = 10
N_set = [(50,10,10), (100,20,10), (150,30,10), (200,40,10), (250,50,10)]
K_set = [2, 4, 6, 8, 10]

"""
Define relaxation term and set an empty dataframe.
"""

k_set = [0,10,50,100,200]
cols = ['PFI'] + [r'$k$ = {0}'.format(k) for k in k_set]
df_GEN = pd.DataFrame(data=0,index=N_set,columns=cols)

for i in range(runs):
    print("Now beginning run number", i)
    data = []
    V_MPFI, V_VFI, V_PFI, X = {}, {}, {}, {}
    for n in N_set:
        d = {}
        X[n] = classes.DuraCons(rho=rho,r=r, eta=eta, iota=iota, theta=theta,
        sigma=sigma, pbar=pbar, bnd=bnd, N=n, K=K_set[N_set.index(n)],
        lambar=lambar, tol=tol, maxiter=maxiter)
        tic = time.time()
        V_PFI[n] = X[n].solve_PFI()
        toc = time.time()
        d['PFI'] = toc-tic
        for k in k_set:
            tic = time.time()
            V_MPFI[n] = X[n].solve_MPFI(k)
            toc = time.time()
            d[r'$k$ = {0}'.format(k)] = toc-tic
            print("k =", k, "n = ", n)
            print("Max absolute difference with PFI:", np.mean(np.abs(V_PFI[n] - V_MPFI[n])))
            print("Max percentage difference with PFI:", np.max(100*np.abs((V_PFI[n] - V_MPFI[n])/V_PFI[n])))
        data.append(d)
    df_GEN = df_GEN + pd.DataFrame(data=data,index=N_set)

df_GEN_ave = df_GEN.round(decimals=2)/runs
df_GEN_ave.index.names = ['Grid size']

destin = '../../figures/durable_table.tex'
with open(destin,'w') as tf:
    tf.write(df_GEN_ave.to_latex(escape=False,column_format='ccccccc'))
