"""
3D LQ problem
"""

import pandas as pd
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import LQ_classes
import timeit
import time

#,(30,30,30),(40,40,40)
data, data_acc, data_acc2 = [], [], []
N_set = [(10,10,10),(20,20,20)]
#k_set = [10,50,100,150]
k_set = []

rho, Q, A = 0.1, 1*np.eye(3), 0.01*np.eye(3)
B, sigma = np.array([.025,.025,.025]).reshape(3,1), 0.4*np.eye(3)
bnd, maxiter, tol = [[0, 10],[0, 10],[0, 10]], 2000, 10**-6
kappa = 3

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
    #V_VFI = X.solve_MPFI(0)
    toc = time.time()
    #d['VFI'] = toc-tic
    print("VFI for grid size", n, "completed in", toc - tic, "seconds")
    #d_acc['VFI'] = np.mean(100*np.abs((X.CF - V_VFI)/X.CF))
    #d_acc2['VFI'] = np.amax(100*np.abs((X.CF - V_VFI)/X.CF))
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

"""
destin = '../../figures/LQ_3D_SD_table.tex'

df = pd.DataFrame(data=data,index=N_set)
cols = df.columns.tolist()
cols = ['PFI', 'VFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

destin = '../../figures/LQ_3D_SDacc_table.tex'

df = pd.DataFrame(data=data_acc,index=N_set)
cols = df.columns.tolist()
cols = ['PFI', 'VFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))
"""

"""
Now generalized case
"""

data, data_acc, data_acc2 = [], [], []
N_set = [(10,10,10),(20,20,20)] #,(30,30,30), (40,40,40)
#k_set = [0,10,50,100,150]
k_set = []

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


"""
Record time and accuracy in a table
"""

destin = '../../figures/LQ_3D_GEN_table.tex'

df = pd.DataFrame(data=data,index=N_set)
cols = df.columns.tolist()
cols = ['PFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

df = pd.DataFrame(data=data_acc,index=N_set)

destin = '../../figures/LQ_3D_GENacc_table.tex'

df = pd.DataFrame(data=data_acc,index=N_set)
cols = df.columns.tolist()
cols = ['PFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

df2 = pd.DataFrame(data=data_acc2,index=N_set)
cols = df2.columns.tolist()
cols = ['PFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df2 = df2[cols].round(decimals=3)
df2.index.names = ['Grid size']
