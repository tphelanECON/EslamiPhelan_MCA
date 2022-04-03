"""
Contour plots for macrofinance example

Convergence more likely (but not assured) if we use a higher relaxation parameter.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import MF_classes

gamma, rho, Pi, rlow = 2., [0.1,0.075], 0.1, 0.0
N, bnd = (120,60), [[0,1], [0.1,0.3]]
mbar, max_iter_eq, pol_maxiter = 4, 1000, 15
theta, sigsigbar = 0.5, 0.15
Delta_y, tol = 10**-4, 10**-6
dt = 10**-9
relax=[0.0]

X = MF_classes.MF_corr(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,
theta=theta,N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt,
mbar=mbar,pol_maxiter=pol_maxiter,relax=relax)

Delta_y = 5*10**-4
Y = MF_classes.MF_corr(rho=rho,gamma=gamma,Pi=Pi,rlow=rlow, sigsigbar=sigsigbar,
theta=theta,N=N,X_bnd=bnd,tol=tol,Delta_y=Delta_y,max_iter_eq=max_iter_eq,dt=dt,
mbar=mbar,pol_maxiter=pol_maxiter,relax=relax)

(r, mux, sigx), time_PFI = X.solve_PFI()
#(rY, muxY, sigxY), time_PFIY = Y.solve_PFI()

"""
See what happens to successive iterations
"""

V_pair = (X.solveV_PFI(0,(r, mux, sigx)), X.solveV_PFI(1,(r, mux, sigx)))
r2, mux2, sigx2 = X.agg_update(V_pair)
V_pair2 = (X.solveV_PFI(0,(r2, mux2, sigx2)), X.solveV_PFI(1,(r2, mux2, sigx2)))
r3, mux3, sigx3 = X.agg_update(V_pair2)

DIFF2 = r - r2, X.XX*(mux - mux2), X.XX*(sigx - sigx2)
DIFF3 = r - r3, X.XX*(mux - mux3), X.XX*(sigx - sigx3)

eps2 = np.amax(np.abs(DIFF2[0][:,1:-1])) + np.amax(np.abs(DIFF2[1][:,1:-1])) + np.amax(np.abs(DIFF2[2][:,1:-1]))
eps3 = np.amax(np.abs(DIFF3[0][:,1:-1])) + np.amax(np.abs(DIFF3[1][:,1:-1])) + np.amax(np.abs(DIFF3[2][:,1:-1]))
print("Difference in updates:",eps2,eps3)

"""
Get value and policy functions, and aggregate consumption and investment
"""

V = (X.solveV_PFI(0,(r, mux, sigx)), X.solveV_PFI(1,(r, mux, sigx)))
cE,kE = X.polupdate(0,V[0],(r, mux, sigx))
cH,kH = X.polupdate(1,V[1],(r, mux, sigx))

c, kEx = cE*X.XX + cH*(1-X.XX), kE*X.XX
cutoff = np.int(np.floor(N[0]/4))
iota = X.Pi - c[cutoff:,1:-1]/kEx[cutoff:,1:-1]

"""
Plots
"""

"""
contourf plots
"""

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[:,1:-1], X.SIGSIG[:,1:-1], r[:,1:-1],500)
fig.colorbar(cp)
plt.title('Interest rate $r$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/MFr.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[cutoff:,1:-1], X.SIGSIG[cutoff:,1:-1], iota,500)
fig.colorbar(cp)
plt.title('Investment function $\iota$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/iota.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[:,1:-1], X.SIGSIG[:,1:-1], X.XX[:,1:-1]*mux[:,1:-1],500)
fig.colorbar(cp)
plt.title('Drift of wealth share $x\mu_x$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/MFmux.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[:,1:-1], X.SIGSIG[:,1:-1], X.XX[:,1:-1]*sigx[:,1:-1],500)
fig.colorbar(cp)
plt.title('Volatility of wealth share $x\sigma_x$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/MFsigx.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[:,1:-1], X.SIGSIG[:,1:-1], V[0][:,1:-1],500)
fig.colorbar(cp)
plt.title('Value function of expert $V_E$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/VE.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(X.XX[:,1:-1], X.SIGSIG[:,1:-1], V[1][:,1:-1],500)
fig.colorbar(cp)
plt.title('Value function of household $V_H$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
destin = '../../figures/VH.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
surface plots
"""

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[:,1:-1], X.SIGSIG[:,1:-1], r[:,1:-1], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Interest rate $r$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/MFr_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[cutoff:,1:-1], X.SIGSIG[cutoff:,1:-1], iota, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Investment function $\iota$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/iota_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[:,1:-1], X.SIGSIG[:,1:-1], X.XX[:,1:-1]*mux[:,1:-1], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Drift of wealth share $x\mu_x$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/MFmux_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[:,1:-1], X.SIGSIG[:,1:-1], X.XX[:,1:-1]*sigx[:,1:-1], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Volatility of wealth share $x\sigma_x$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/MFsigx_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[:,1:-1], X.SIGSIG[:,1:-1], V[0][:,1:-1], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Value function of expert $V_E$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/VE_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.XX[:,1:-1], X.SIGSIG[:,1:-1], V[1][:,1:-1], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
plt.title('Value function of household $V_H$')
ax.set_xlabel('Expert wealth share $x$', fontsize=13)
ax.set_ylabel('Volatility $\sigma$', fontsize=13)
ax.view_init(30, 45)
destin = '../../figures/VH_surf.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
