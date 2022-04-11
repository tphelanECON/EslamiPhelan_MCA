"""
Various partial checks on the algorithm:

Check 1: recovery of log utility values for gamma near 1.
Check 2: recovery of known boundary values in absence of mean reversion.
Check 3: check "constant dt" and "variable dt" algorithms agree (approximately)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.optimize as scopt
from scipy.sparse import linalg
from scipy.sparse import diags
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import time, itertools, math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import MF_classes

"""
Check 1: recovery of log utility values for gamma near 1.
"""

print("Performing first check")

rho, Pi, rlow = [0.1,0.075], 0.05, 0.001
N, bnd = (120,60), [[0,1], [0.1,0.3]]
mbar, max_iter_eq = 4, 1000
theta, sigsigbar= 0.5, 0.15
dt, Delta_y, tol = 10**-8, 10**-4, 10**-6
data, gamma_set = [], [0.9,0.99,0.995]
for gamma in gamma_set:
    print(gamma)
    d = {}
    X = MF_classes.MF_corr(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,theta=theta, \
    N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt,mbar=mbar)
    tic = time.time()
    (r, mux, sigx), toctic = X.solve_PFI()
    toc = time.time()
    print("Time taken:", toc-tic)
    r_log, mux_log, sigx_log = X.log_quant()
    diff = r - r_log, X.XX*(mux - mux_log), X.XX*(sigx - sigx_log)
    d[r'$r$'] = np.max(np.abs(diff[0]))
    d[r'$x\mu_x$'] = np.max(np.abs(diff[1]))
    d[r'$x\sigma_x$'] = np.max(np.abs(diff[2]))
    data.append(d)

df = pd.DataFrame(data=data,index=gamma_set)
cols = df.columns.tolist()
df = df[cols]
df.index.names = ['$\gamma$']
destin = '../../figures/log_table_check.tex'

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='cccc'))

"""
Check 2: recovery of known boundary values in absence of mean reversion.
Use class constructor with independent noise, since high correlation
algorithm breaks down when sigsigbar vanishes.

In the following, the differences should be zero when x=0,1.
"""

print("Performing second check")

Pi = 0.1
gamma, theta, sigsigbar = 2., 0.0, 0.0
X = MF_classes.MF_ind(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,theta=theta, \
N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt)

tic = time.time()
(r, mux, sigx), toctic = X.solve_PFI()
toc = time.time()

V_pair = (X.solveV_PFI(0,(r, mux, sigx)), X.solveV_PFI(1,(r, mux, sigx)))
V_pair_bnd = (X.V_bnd_lin(0), X.V_bnd_lin(1))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX, X.SIGSIG, V_pair[1]**(1/(1-X.gamma)) - V_pair_bnd[1]**(1/(1-X.gamma)), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_xlabel('Wealth share $x$', fontsize=13)
ax.set_ylabel('Exogenous uncertainty $\sigma$', fontsize=13)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX, X.SIGSIG, V_pair[0]**(1/(1-X.gamma)) - V_pair_bnd[0]**(1/(1-X.gamma)), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_xlabel('Wealth share $x$', fontsize=13)
ax.set_ylabel('Exogenous uncertainty $\sigma$', fontsize=13)
plt.show()

"""
Check 3: check "constant dt" and "variable dt" algorithms agree (approximately)
"""

print("Performing third check")

pbar, mbar, max_iter_eq = 10**-6, 4, 1000
theta, sigsigbar = 1.0,0.2
X = MF_classes.MF_corr(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,theta=theta, \
N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt,mbar=mbar)

Y = MF_classes.MF_corr_var_dt(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,theta=theta, \
N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,pbar=pbar,mbar=mbar)

tic = time.time()
(r, mux, sigx), toctic = X.solve_PFI()
rY, muxY, sigxY = Y.solve_PFI()
toc = time.time()

"""
Following should be small
"""

print("Average absolute differences in interest rates:", np.mean(np.abs(r-rY)))
print("Average absolute differences in xmux:", np.mean(np.abs(X.XX*mux-Y.XX*muxY)))
print("Average absolute differences in xsigx:", np.mean(np.abs(X.XX*sigx-Y.XX*sigxY)))
