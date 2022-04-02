"""
Create plots for the durable good example used in the paper.
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

data = []
n, K, k = (200,40,10), 8, 150

rho, r, eta, iota = -np.log(0.93), 0.06, 1/0.77-1, 0.01
theta, sigma = -np.log(0.95), np.sqrt(-2*np.log(0.95))*0.2
pbar, lambar = 1, 52
lambar2 = 365

bnd, maxiter, tol = [[0,100],[-0.8,0.8],[0,12]], 5000, 10**-6

X = classes.DuraCons(rho=rho,r=r, eta=eta, iota=iota, theta=theta, sigma=sigma,
pbar=pbar, bnd=bnd,N=n, K=K, lambar=lambar,tol=tol, maxiter=maxiter)
tic = time.time()
V = X.solve_MPFI(k)
toc = time.time()
print(toc-tic)

Y = classes.DuraCons(rho=rho,r=r, eta=eta, iota=iota, theta=theta, sigma=sigma,
pbar=pbar, bnd=bnd,N=n, K=K, lambar=lambar2, tol=tol, maxiter=maxiter)
tic = time.time()
VY = Y.solve_MPFI(k)
toc = time.time()
print(toc-tic)

print("Difference in V when lambda changes from", lambar, "to", lambar2)
print("Supremum norm:", np.max(np.abs(V-VY)))
print("$L_1$ norm:", np.mean(np.abs(V-VY)))

"""
Want to plot consumption and durable good policy functions.
"""

c, lam = X.polupdate(V)
cY, lamY = Y.polupdate(VY)
D = 5

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.xx[0][0:-1,:,D], np.exp(X.xx[1][0:-1,:,D]), c[0:-1,:,D] - cY[0:-1,:,D], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.view_init(30, 225)
ax.set_title('')
ax.set_xlabel('$a$')
ax.set_ylabel('$y$')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.xx[0][0:-1,:,6], X.xx[1][0:-1,:,6], lam[0:-1,:,6]/X.lambar - lamY[0:-1,:,6]/Y.lambar, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.view_init(30, 225)
ax.set_title('')
ax.set_xlabel('$a$')
ax.set_ylabel('$y$')
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.xx[0][0:-1,:,D], np.exp(X.xx[1][0:-1,:,D]), c[0:-1,:,D],500)
fig.colorbar(cp)
ax.set_xlabel('Assets $a$', fontsize=13)
ax.set_ylabel('Income $y$', fontsize=13)
ax.set_title('Consumption for $D$ = {0}'.format(D), fontsize=13)
destin = '../../figures/duraslice_C.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.xx[0][0:-1,:,D], np.exp(X.xx[1][0:-1,:,D]), lam[0:-1,:,D]/X.lambar, 100)
fig.colorbar(cp)
ax.set_xlabel('Assets $a$', fontsize=13)
ax.set_ylabel('Income $y$', fontsize=13)
ax.set_title('Durable consumption policy for $D$ = {0}'.format(D), fontsize=13)
destin = '../../figures/duraslice_lam.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.xx[0][0:-1,:,D+1], np.exp(X.xx[1][0:-1,:,D+1]), lam[0:-1,:,D+1]/X.lambar, 100)
fig.colorbar(cp)
ax.set_xlabel('Assets $a$', fontsize=13)
ax.set_ylabel('Income $y$', fontsize=13)
ax.set_title('Durable consumption policy for $D$ = {0}'.format(D+1), fontsize=13)
destin = '../../figures/duraslice_lam2.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.xx[0][0:-1,:,D+2], np.exp(X.xx[1][0:-1,:,D+2]), lam[0:-1,:,D+2]/X.lambar, 100)
fig.colorbar(cp)
ax.set_xlabel('Assets $a$', fontsize=13)
ax.set_ylabel('Income $y$', fontsize=13)
ax.set_title('Durable consumption policy for $D$ = {0}'.format(D+2), fontsize=13)
destin = '../../figures/duraslice_lam3.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
