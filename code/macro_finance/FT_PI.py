"""
Comparison of false transient and policy iteration algorithms.

solve_FT takes argument check = 0 or 1:
    * check = 1: solve_FT begins at values found in solve_PFI.
    * check = 0: begins at logarithmic quantities.
"""

import numpy as np
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
