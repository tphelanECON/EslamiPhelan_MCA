"""
Independent case.

Creates figures but does not save them as they are not used in the paper.
"""

import numpy as np
import pandas as pd
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
rho, Pi = [0.1,0.075], 0.1
N, bnd = (120,60), [[0,1],[0.1,0.3]]
gamma, theta, sigsigbar = 2., 1., 0.2
Delta_y, max_iter_eq = 10**-2, 1000
dt=10**(-8)
rlow = 0.001
"""

gamma, rho, Pi, rlow = 2., [0.08,0.06], 0.1, 0.0
N, bnd = (160,80), [[0,1], [0.1,0.3]]
mbar, max_iter_eq = 4, 1000
theta, sigsigbar = 0.5, 0.5
dt, Delta_y, tol = 5*10**-8, 2.0*10**-4, 10**-6

X = MF_classes.MF_ind(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,theta=theta, \
N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt)

(r, mux, sigx), time_PFI = X.solve_PFI()
print("Time taken:", time_PFI)

V_pair = (X.solveV_PFI(0,(r, mux, sigx)), X.solveV_PFI(1,(r, mux, sigx)))
V_pair_bnd = (X.V_bnd_lin(0), X.V_bnd_lin(1))
(cE,kE) = X.polupdate(0,V_pair[0],(r, mux, sigx))
(cH,kH) = X.polupdate(1,V_pair[1],(r, mux, sigx))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX, X.SIGSIG, kE*X.XX, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma$')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX, X.SIGSIG, cE - X.rho[0], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma$')
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX, X.SIGSIG, r, 500)
fig.colorbar(cp)
plt.title('Interest rate')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Exogenous uncertainty $\sigma$', fontsize=13)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX, X.SIGSIG, X.XX*mux, 500)
fig.colorbar(cp)
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Exogenous uncertainty $\sigma$', fontsize=13)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX, X.SIGSIG, X.XX*sigx, 500)
fig.colorbar(cp)
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Exogenous uncertainty $\sigma$', fontsize=13)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX, X.SIGSIG, r, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.view_init(30, 45)
ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma$')
plt.show()
