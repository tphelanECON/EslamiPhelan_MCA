"""
Income-fluctuation problem in three dimensions.

"""

import numpy as np
import pandas as pd
import time, classes

"""
Declare parameters.

Annual autocorrelation 0.95, so theta = -np.log(0.95).
Stat dist for OU process: Gaussian with variance sigma**2/(2*theta).
Achdou choose nu = 0.2 and and hence sigma = np.sqrt(2*theta)*nu = 0.064.
"""

rho, r, gamma, kappa = 1/0.95-1, 0.03, 2.0, 2.0
nu = 0.2
theta, sigma = [-np.log(0.95),-np.log(0.95)], [np.sqrt(-2*np.log(0.95))*nu,np.sqrt(-2*np.log(0.95))*nu]
bnd, maxiter, tol, mono_tol = [[0, 170], [-4*nu, 4*nu], [-4*nu, 4*nu]], 4000, 10**-6, 10**-6

"""
Define number of runs and grid sizes.
"""

runs = 10
N_set = [(45,15,15), (60,20,20), (75,25,25), (90,30,30)]

"""
Define relaxation term and set an empty dataframe.
"""

k_set = [10, 50, 100, 200]
cols = ['PFI', 'VFI'] + [r'$k$ = {0}'.format(k) for k in k_set]
df = pd.DataFrame(data=0,index=N_set,columns=cols)

for i in range(runs):
    print("Now beginning run number", i)
    data = []
    for n in N_set:
        d = {}
        X = classes.IFP_3D(rho=rho,r=r,gamma=gamma,theta=theta,sigma=sigma,
        bnd=bnd,N=n,tol=tol,mono_tol=mono_tol,maxiter=maxiter,kappa=kappa)
        tic = time.time()
        VX_PFI = X.solve_PFI()
        toc = time.time()
        d['PFI'] = toc-tic
        print("PFI for grid size", n, "completed")
        tic = time.time()
        VX_VFI = X.solve_MPFI(0)
        toc = time.time()
        d['VFI'] = toc-tic
        print("VFI for grid size", n, "completed")
        for k in k_set:
            tic = time.time()
            VX = X.solve_MPFI(k)
            toc = time.time()
            d[r'$k$ = {0}'.format(k)] = toc-tic
            print("k =", k, "n = ", n, "completed")
            print("Max absolute difference between PFI and MPFI", k, ":", np.max(np.abs(VX - VX_PFI)))
            print("Max percentage difference with PFI and MPFI", k, ":", np.max(100*np.abs((VX - VX_PFI)/VX_PFI)))
        data.append(d)
    df = df + pd.DataFrame(data=data,index=N_set,columns=cols)

df_ave = df.round(decimals=2)/runs
df_ave.index.names = ['Grid size']

destin = '../../figures/IFP_3D_table.tex'
with open(destin,'w') as tf:
    tf.write(df_ave.to_latex(escape=False,column_format='ccccccc'))

"""
Generalized algorithm. Do not label k=0 'VFI'. No need to redefine
"""

k_set = [0, 10, 50, 100, 200]
cols = ['PFI'] + [r'$k$ = {0}'.format(k) for k in k_set]
df_GEN = pd.DataFrame(data=0,index=N_set,columns=cols)

for i in range(runs):
    print("Now beginning run number", i)
    data = []
    for n in N_set:
        d = {}
        Y = classes.GIFP_3D(rho=rho,r=r,gamma=gamma,theta=theta,sigma=sigma,
        bnd=bnd,N=n,tol=tol,maxiter=maxiter)
        tic = time.time()
        VY_PFI = Y.solve_PFI()
        toc = time.time()
        d['PFI'] = toc-tic
        print("PFI for grid size", n, "completed")
        for k in k_set:
            tic = time.time()
            VY = Y.solve_MPFI(k)
            toc = time.time()
            d[r'$k$ = {0}'.format(k)] = toc-tic
            print("k =", k, "n = ", n)
            print("Max absolute difference with GPFI:", np.max(np.abs(VY - VY_PFI)))
            print("Max percentage difference with GPFI:", np.max(100*np.abs((VY - VY_PFI)/VY_PFI)))
        data.append(d)
    df_GEN = df_GEN + pd.DataFrame(data=data,index=N_set,columns=cols)

df_GEN_ave = df_GEN.round(decimals=2)/runs
df_GEN_ave.index.names = ['Grid size']

destin = '../../figures/GIFP_3D_table.tex'
with open(destin,'w') as tf:
    tf.write(df_GEN_ave.to_latex(escape=False,column_format='ccccccc'))

"""
MPFI and GPFI need not return the same value (they are solving different problems
as the timesteps are different). However, reassuringly, they are close.
"""

print("Max absolute difference across algorithms:", np.max(np.abs(VX_PFI - VY_PFI)))
print("Max percentage difference across algorithms:", np.max(100*np.abs((VX_PFI - VY_PFI)/VY_PFI)))

print("Mean absolute difference across algorithms:", np.mean(np.abs(VX_PFI - VY_PFI)))
print("Mean percentage difference across algorithms:", np.mean(100*np.abs((VX_PFI - VY_PFI)/VY_PFI)))
