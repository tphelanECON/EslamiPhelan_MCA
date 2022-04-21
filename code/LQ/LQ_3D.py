"""
3D LQ problem
"""

import pandas as pd
import numpy as np
import time, LQ_classes

rho, Q, A = 0.1, 1*np.eye(3), 0.01*np.eye(3)
B, sigma = np.array([.025,.025,.025]).reshape(3,1), 0.4*np.eye(3)
bnd, maxiter, tol = [[0, 10], [0, 10], [0, 10]], 2000, 10**-6
kappa = 3

"""
Define number of runs and grid sizes.
"""

runs = 10
N_set = [(10,10,10), (20,20,20), (30,30,30), (40,40,40)]

"""
Define relaxation term and set an empty dataframe.
"""

k_set = [10, 50, 100, 200]
cols = ['PFI', 'VFI'] + [r'$k$ = {0}'.format(k) for k in k_set]
df = pd.DataFrame(data=0,index=N_set,columns=cols)
df_acc = pd.DataFrame(data=0,index=N_set,columns=cols)
df_acc2 = pd.DataFrame(data=0,index=N_set,columns=cols)

for i in range(runs):
    print("Now beginning run number", i)
    data, data_acc, data_acc2 = [], [], []
    for n in N_set:
        d, d_acc, d_acc2 = {}, {}, {}
        X = LQ_classes.LQ_3D_SD(rho=rho,Q=Q,A=A,B=B,sigma=sigma,N=n,bnd=bnd,tol=tol,maxiter=maxiter,kappa=kappa)
        tic = time.time()
        V_PFI = X.solve_PFI()
        toc = time.time()
        d['PFI'] = toc-tic
        print("PFI for grid size", n, "completed in", toc - tic, "seconds")
        d_acc['PFI'] = np.mean(100*np.abs((X.CF - V_PFI)/X.CF))
        d_acc2['PFI'] = np.amax(100*np.abs((X.CF - V_PFI)/X.CF))
        tic = time.time()
        V_VFI = X.solve_MPFI(0)
        toc = time.time()
        d['VFI'] = toc-tic
        print("VFI for grid size", n, "completed in", toc - tic, "seconds")
        d_acc['VFI'] = np.mean(100*np.abs((X.CF - V_VFI)/X.CF))
        d_acc2['VFI'] = np.amax(100*np.abs((X.CF - V_VFI)/X.CF))
        for k in k_set:
            tic = time.time()
            V_MPFI = X.solve_MPFI(k)
            toc = time.time()
            d[r'$k$ = {0}'.format(k)] = toc-tic
            d_acc[r'$k$ = {0}'.format(k)] = np.mean(100*np.abs((X.CF - V_MPFI)/X.CF))
            d_acc2[r'$k$ = {0}'.format(k)] = np.max(100*np.abs((X.CF - V_MPFI)/X.CF))
            print("MPFI for grid size", n, "and relaxation", k, "completed in", toc - tic, "seconds")
            print("Mean absolute percentage error:", np.mean(100*np.abs((X.CF - V_MPFI)/X.CF)))
            print("Max absolute percentage error:", np.amax(100*np.abs((X.CF - V_MPFI)/X.CF)))
        data.append(d)
        data_acc.append(d_acc)
        data_acc2.append(d_acc2)
    df = df + pd.DataFrame(data=data,index=N_set)
    df_acc = df_acc + pd.DataFrame(data=data_acc,index=N_set)
    df_acc2 = df_acc2 + pd.DataFrame(data=data_acc2,index=N_set)

df_ave = df.round(decimals=3)/runs
df_ave.index.names = ['Grid size']

destin = '../../figures/LQ_3D_SD_table.tex'
with open(destin,'w') as tf:
    tf.write(df_ave.to_latex(escape=False,column_format='ccccccc'))

df_acc_ave = df_acc.round(decimals=3)/runs
df_acc_ave.index.names = ['Grid size']

destin = '../../figures/LQ_3D_SDacc_table.tex'
with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

"""
Now generalized case
"""

k_set = [0, 10, 50, 100, 200]
cols = ['PFI'] + [r'$k$ = {0}'.format(k) for k in k_set]
df_GEN = pd.DataFrame(data=0,index=N_set,columns=cols)
df_GEN_acc = pd.DataFrame(data=0,index=N_set,columns=cols)
df_GEN_acc2 = pd.DataFrame(data=0,index=N_set,columns=cols)

for i in range(runs):
    print("Now beginning run number", i)
    data, data_acc, data_acc2 = [], [], []
    for n in N_set:
        d, d_acc, d_acc2 = {}, {}, {}
        X = LQ_classes.LQ_3D_GEN(rho=rho,Q=Q,A=A,B=B,sigma=sigma,N=n,bnd=bnd,tol=tol,maxiter=maxiter)
        tic = time.time()
        V_PFI = X.solve_PFI()
        toc = time.time()
        d['PFI'] = toc-tic
        print("PFI for grid size", n, "completed in", toc-tic, "seconds")
        d_acc['PFI'] = np.mean(100*np.abs((X.CF - V_PFI)/X.CF))
        d_acc2['PFI'] = np.amax(100*np.abs((X.CF - V_PFI)/X.CF))
        for k in k_set:
            tic = time.time()
            V_MPFI = X.solve_MPFI(k)
            toc = time.time()
            d[r'$k$ = {0}'.format(k)] = toc-tic
            d_acc[r'$k$ = {0}'.format(k)] = np.mean(100*np.abs((X.CF - V_MPFI)/X.CF))
            d_acc2[r'$k$ = {0}'.format(k)] = np.max(100*np.abs((X.CF - V_MPFI)/X.CF))
            print("GMPFI for grid size", n, "and relaxation", k, "completed in", toc - tic, "seconds")
            print("Mean absolute percentage error:", np.mean(100*np.abs((X.CF - V_MPFI)/X.CF)))
            print("Max absolute percentage error:", np.amax(100*np.abs((X.CF - V_MPFI)/X.CF)))
        data.append(d)
        data_acc.append(d_acc)
        data_acc2.append(d_acc2)
    df_GEN = df_GEN + pd.DataFrame(data=data,index=N_set)
    df_GEN_acc = df_GEN_acc + pd.DataFrame(data=data_acc,index=N_set)
    df_GEN_acc2 = df_GEN_acc2 + pd.DataFrame(data=data_acc2,index=N_set)

"""
Record time and accuracy in a table
"""

df_GEN_ave = df_GEN.round(decimals=3)/runs
df_GEN_ave.index.names = ['Grid size']

destin = '../../figures/LQ_3D_GEN_table.tex'
with open(destin,'w') as tf:
    tf.write(df_GEN_ave.to_latex(escape=False,column_format='ccccccc'))

df_GEN_acc_ave = df_GEN_acc.round(decimals=3)/runs
df_GEN_acc_ave.index.names = ['Grid size']

destin = '../../figures/LQ_3D_GENacc_table.tex'
with open(destin,'w') as tf:
    tf.write(df_GEN_acc_ave.to_latex(escape=False,column_format='ccccccc'))
