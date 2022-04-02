"""
Income-fluctuation problem in three dimensions.

Need to make sure all paths in the following are relative.
"""

import numpy as np
import pandas as pd
import time
import scipy
import scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import classes

"""
Declare parameters.

Annual autocorrelation 0.95, so theta = -np.log(0.95).
Stat dist for OU process: Gaussian with variance sigma**2/(2*theta).
Achdou choose nu = 0.2 and and hence sigma = np.sqrt(2*theta)*nu = 0.064.
"""

rho, r, gamma = 1/0.95-1, 0.03, 2.0
theta, sigma = [-np.log(0.95),-np.log(0.95)], [np.sqrt(-2*np.log(0.95))*0.2,np.sqrt(-2*np.log(0.95))*0.2]
bnd, maxiter, tol, mono_tol = [[0,60],[-0.8,0.8],[-0.8,0.8]], 4000, 10**-6, 10**-10
kappa = 2.

#, (60,20,20), (75,25,25)
#, (45,15,15)
VF_3D = {}
data = []
N_set = [(30,10,10), (45,15,15), (60,20,20)]
k_set = []
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
    #VX_VFI = X.solve_MPFI(0)
    toc = time.time()
    d['VFI'] = toc-tic
    print("VFI for grid size", n, "completed")
    for k in k_set:
        tic = time.time()
        VX = X.solve_MPFI(k)
        toc = time.time()
        d[r'$k$ = {0}'.format(k)] = toc-tic
        #print("k =", k, "n = ", n, "completed")
        #print("Max absolute difference between PFI and MPFI", k, ":", np.max(np.abs(VX - VX_PFI)))
        #print("Max percentage difference with PFI and MPFI", k, ":", np.max(100*np.abs((VX - VX_PFI)/VX_PFI)))
    data.append(d)

"""
destin = '../../figures/IFP_3D_table.tex'

df = pd.DataFrame(data=data,index=N_set)
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
Generalized algorithm. Do not label k=0 'VFI'.
"""

k_set = []
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

"""
MPFI and GPFI need not return exact same value as Dt = 0 in latter. However,
reassuringly, they are close (approx. 1% maximum difference for larger grids).
"""

print("Max absolute difference across algorithms:", np.max(np.abs(VX_PFI - VY_PFI)))
print("Max percentage difference across algorithms:", np.max(100*np.abs((VX_PFI - VY_PFI)/VY_PFI)))

print("Mean absolute difference across algorithms:", np.mean(np.abs(VX_PFI - VY_PFI)))
print("Mean percentage difference across algorithms:", np.mean(100*np.abs((VX_PFI - VY_PFI)/VY_PFI)))

"""
destin = '../../figures/GIFP_3D_table.tex'

df = pd.DataFrame(data=data,index=N_set)
cols = df.columns.tolist()
cols = ['PFI']
for k in k_set:
    cols.append(r'$k$ = {0}'.format(k))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))
"""

"""
V = Y.solve_PFI()
c = Y.polupdate(V)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y.xx[0][:,:,6], Y.xx[1][:,:,6], c[:,:,6], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.view_init(30, 225)
ax.set_title('')
ax.set_xlabel('$a$')
ax.set_ylabel('$y$')
plt.show()

cX = X.polupdate(VX_PFI)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.xx[0][:,:,6], X.xx[1][:,:,6], VX_PFI[:,:,6], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.view_init(30, 225)
ax.set_title('')
ax.set_xlabel('$a$')
ax.set_ylabel('$y$')
plt.show()
"""
