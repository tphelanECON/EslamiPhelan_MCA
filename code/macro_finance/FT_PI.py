"""
Comparison of false transient and policy iteration algorithms.

solve_FT takes argument check = 0 or 1:
    * check = 1: solve_FT begins at values found in solve_PFI.
    * check = 0: begins at logarithmic quantities.

"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as scopt
from scipy.sparse import linalg
from scipy.sparse import diags
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import time, math, itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import MF_classes

"""
Set common parameters
"""

gamma, rho, Pi, rlow = 2., [0.1,0.075], 0.1, 0.0
N, bnd = (40,20), [[0,1], [0.1,0.3]]
mbar, max_iter_eq, pol_maxiter = 4, 12000, 12
theta, sigsigbar = 0.5, 0.15
Delta_y, tol = 10**-1, 10**-6
relax = [0.0]
dt = [1*10**-3, 2*10**-3, 3*10**-3]
X, time_PFI, time_FT, time_FT2 = {}, {}, {}, {}
agg_PFI, agg_FT, agg_FT2 = {}, {}, {}
difference, iterations = {}, {}
difference2, iterations2 = {}, {}

for i in range(len(dt)):
    X[i] = MF_classes.MF_corr(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar, \
    theta=theta,N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt[i], \
    mbar=mbar,pol_maxiter=pol_maxiter,relax=relax)
    agg_PFI[i], time_PFI[i] = X[i].solve_PFI()
    agg_FT[i], (difference[i], iterations[i]), time_FT[i] = X[i].solve_FT(0)
    agg_FT2[i], (difference2[i], iterations2[i]), time_FT2[i] = X[i].solve_FT(1)

fig,ax = plt.subplots()
for i in range(len(dt)):
    ax.plot(iterations[i], np.log10(difference[i]), label="$\Delta_t$ = {0}".format(np.round(10**3*dt[i],2)) +  " x $10^{-3}$", linewidth=2)
plt.title('Beginning at logarithmic quantities')
ax.set_ylabel('log$_{10}$(E)')
ax.set_xlabel('Number of iterations')
ax.legend(loc = 'upper right')
destin = '../../figures/diff_MF.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig,ax = plt.subplots()
for i in range(len(dt)):
    ax.plot(iterations2[i], np.log10(difference2[i]), label="$\Delta_t$ = {0}".format(np.round(10**3*dt[i],2)) +  " x $10^{-3}$", linewidth=2)
plt.title('Beginning at quantities found with policy iteration')
ax.set_ylabel('log$_{10}$(E)')
ax.set_xlabel('Number of iterations')
ax.legend(loc = 'upper right')
destin = '../../figures/diff_MF2.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

V_pair = (X[i].solveV_PFI(0,agg_PFI[i]), X[i].solveV_PFI(1,agg_PFI[i]))
r, mux, sigx = X[i].agg_update(V_pair)
V_pair2 = (X[i].solveV_PFI(0,(r, mux, sigx)), X[i].solveV_PFI(1,(r, mux, sigx)))
r2, mux2, sigx2 = X[i].agg_update(V_pair2)
V_pair3 = (X[i].solveV_PFI(0,(r2, mux2, sigx2)), X[i].solveV_PFI(1,(r2, mux2, sigx2)))
r3, mux3, sigx3 = X[i].agg_update(V_pair3)

DIFF = r - agg_PFI[i][0], X[i].XX*(mux - agg_PFI[i][1]), X[i].XX*(sigx - agg_PFI[i][2])
DIFF2 = r2 - agg_PFI[i][0], X[i].XX*(mux2 - agg_PFI[i][1]), X[i].XX*(sigx2 - agg_PFI[i][2])
DIFF3 = r3 - agg_PFI[i][0], X[i].XX*(mux3 - agg_PFI[i][1]), X[i].XX*(sigx3 - agg_PFI[i][2])

eps = np.amax(np.abs(DIFF[0][:,1:-1])) + np.amax(np.abs(DIFF[1][:,1:-1])) + np.amax(np.abs(DIFF[2][:,1:-1]))
eps2 = np.amax(np.abs(DIFF2[0][:,1:-1])) + np.amax(np.abs(DIFF2[1][:,1:-1])) + np.amax(np.abs(DIFF2[2][:,1:-1]))
eps3 = np.amax(np.abs(DIFF3[0][:,1:-1])) + np.amax(np.abs(DIFF3[1][:,1:-1])) + np.amax(np.abs(DIFF3[2][:,1:-1]))
print(eps,eps2,eps3)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X[i].XX[:,1:-1], X[i].SIGSIG[:,1:-1], mux[:,1:-1] - mux3[:,1:-1], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_title('')
ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma$')
plt.show()
