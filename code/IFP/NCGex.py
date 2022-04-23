"""
Growth model with non-concave technology.
"""

import numpy as np
import scipy as sp
import math, time
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.optimize as scopt
from scipy.sparse import linalg

"""
Non-concave growth problem. The slope and sigma parameters create a function
for volatility that vanishes near the endpoints of the grid.
WP was slightly different and used self.sigma = self.sigbar*(self.bnd[1]>self.kgrid)*(self.bnd[0]<self.kgrid)
"""

class NCG(object):
    def __init__(self,rho=1,sigbar=0.1,N=1000,A=1,B=5,kstar=5.,delta=0.05,
    alpha=0.5,Dt=10**-8,bnd=[1,20],tol=10**-6,slope=2,maxiter=50):
        self.A, self.B, self.kstar, self.sigbar = A, B, kstar, sigbar
        self.rho, self.delta, self.alpha = rho, delta, alpha
        self.N, self.bnd, self.tol = N, bnd, tol
        self.Delta, self.Dt = (self.bnd[1]-self.bnd[0])/self.N, Dt
        self.kgrid = np.linspace(self.bnd[0],self.bnd[1],self.N+1)
        self.slope = 0.05
        smooth_vanish = np.minimum(self.slope*(self.kgrid - self.bnd[0])**2, self.slope*(self.bnd[1]-self.kgrid)**2)
        self.sigma = np.minimum(self.sigbar,self.slope*(self.bnd[1]-self.kgrid)**0.75)*(self.bnd[0]<self.kgrid)
        #self.sigma = self.sigbar*(self.bnd[1]>self.kgrid)*(self.bnd[0]<self.kgrid)
        self.c0 = self.f(self.kgrid)-self.delta*self.kgrid
        self.maxiter = maxiter

    def f(self,k):
        return np.maximum(self.A*k**self.alpha,self.B*(np.maximum(0,k-self.kstar)**self.alpha))

    def Vupdate(self,c):
        P_func = lambda A,B,C : sp.coo_matrix((C,(A,B)),shape=(self.N+1,self.N+1))
        mu, d = self.f(self.kgrid) - self.delta*self.kgrid - c, self.Dt/self.Delta**2
        up = d*(self.sigma**2*self.kgrid**2/2 + self.Delta*np.maximum(0,mu))
        down = d*(self.sigma**2*self.kgrid**2/2 + self.Delta*np.maximum(0,-mu))
        if up[-1] > 0 or down[0] > 0:
            print("Error: state not moving in right direction at boundary points")
        if np.min(1-up-down) < 0:
            print("Error: some probabilities negative", np.min(1-up-down) )
        P = P_func(range(self.N+1),range(self.N+1),1-up-down) \
        + P_func(range(self.N),range(1,self.N+1),up[:-1]) \
        + P_func(range(1,self.N+1),range(self.N),down[1:])
        b = self.Dt*self.rho*np.log(c)
        B = sp.eye(self.N+1)-np.exp(-self.rho*self.Dt)*P
        return sp.linalg.spsolve(B,b)

    def polupdate(self,V):
        Fdiff = np.append(np.diff(V)/self.Delta,-10**6)
        Bdiff = np.roll(Fdiff,1)
        #objective function:
        H = lambda c: self.rho*np.log(c) + np.exp(-self.rho*self.Dt) \
        *(np.maximum(self.f(self.kgrid)-c-self.delta*self.kgrid,0)*Fdiff \
        - np.maximum(-(self.f(self.kgrid)-c-self.delta*self.kgrid), 0)*Bdiff)
        #two candidate consumption functions:
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum(self.rho*np.exp(self.rho*self.Dt)/Fdiff,self.c0)
            chigh = np.maximum(self.rho*np.exp(self.rho*self.Dt)/Bdiff,self.c0)
        clow[Fdiff<=0], chigh[Bdiff<=0] = self.c0[Fdiff<=0], self.c0[Bdiff<=0]
        I = np.argmin(np.vstack([-H(self.c0),-H(clow), -H(chigh)]), axis=0)
        C = (I==0)*self.c0 + (I==1)*clow + (I==2)*chigh
        #C[0], C[-1] = np.minimum(C[0],clow[0]), np.maximum(chigh[-1],C[-1])
        return C

    def solveV(self,V0):
        eps, i = 1, 1
        V = V0
        while i < self.maxiter and eps > self.tol:
            V1 = self.Vupdate(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < 0:
                nonmon = V1-V<0
                print("Failure of monotonicity at:", len(self.kgrid[nonmon]), "points")
                print("Average magnitude of failure:", np.mean(np.abs(V1[nonmon] - V[nonmon])))
            i = i+1
            V = V1
            print(eps)
        if i < self.maxiter:
            print("Policy function converged in:", i, "iterations")
        else:
            print("Policy function did not converge. Difference:", eps)
        return V

class NCG_zero_dt(object):
    def __init__(self,rho=1,sigbar=0.1,N=1000,A=1,B=5,kstar=5.,delta=0.05,
    alpha=0.5,bnd=[1,20],tol=10**-6,slope=2,maxiter=50):
        self.A, self.B, self.kstar, self.sigbar = A, B, kstar, sigbar
        self.rho, self.delta, self.alpha = rho, delta, alpha
        self.N, self.bnd, self.tol, self.maxiter = N, bnd, tol, maxiter
        self.Delta = (self.bnd[1]-self.bnd[0])/self.N
        self.kgrid = np.linspace(self.bnd[0],self.bnd[1],self.N+1)
        self.slope = 0.05
        smooth_vanish = np.minimum(self.slope*(self.kgrid - self.bnd[0])**2, self.slope*(self.bnd[1]-self.kgrid)**2)
        self.sigma = np.minimum(self.sigbar,self.slope*(self.bnd[1]-self.kgrid)**0.75)*(self.bnd[0]<self.kgrid)
        self.c0 = self.f(self.kgrid)-self.delta*self.kgrid

    def f(self,k):
        return np.maximum(self.A*k**self.alpha,self.B*(np.maximum(0,k-self.kstar)**self.alpha))

    def Vupdate(self,c):
        P_func = lambda A,B,C : sp.coo_matrix((C,(A,B)),shape=(self.N+1,self.N+1))
        mu, d = self.f(self.kgrid) - self.delta*self.kgrid - c, 1/self.Delta**2
        up = d*(self.sigma**2*self.kgrid**2/2 + self.Delta*np.maximum(0,mu))
        down = d*(self.sigma**2*self.kgrid**2/2 + self.Delta*np.maximum(0,-mu))
        if up[-1] > 0 or down[0] > 0:
            print("Error: state not moving in right direction at boundary points")
        T = P_func(range(self.N+1),range(self.N+1),-self.rho-up-down) \
        + P_func(range(self.N),range(1,self.N+1),up[:-1]) \
        + P_func(range(1,self.N+1),range(self.N),down[1:])
        b = self.rho*np.log(c)
        T,b = 10**3*T,10**3*b
        return sp.linalg.spsolve(-T,b)

    def polupdate(self,V):
        Fdiff = np.append(np.diff(V)/self.Delta,-10**6)
        Bdiff = np.roll(Fdiff,1)
        #objective function:
        H = lambda c: self.rho*np.log(c) \
        + (np.maximum(self.f(self.kgrid)-c-self.delta*self.kgrid,0)*Fdiff \
        - np.maximum(-(self.f(self.kgrid)-c-self.delta*self.kgrid), 0)*Bdiff)
        #two candidate consumption functions:
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum(self.rho/Fdiff,self.c0)
            chigh = np.maximum(self.rho/Bdiff,self.c0)
        clow[Fdiff<=0], chigh[Bdiff<=0] = self.c0[Fdiff<=0], self.c0[Bdiff<=0]
        I = np.argmin(np.vstack([-H(self.c0),-H(clow), -H(chigh)]), axis=0)
        C = (I==0)*self.c0 + (I==1)*clow + (I==2)*chigh
        return C

    def solveV(self,V0):
        eps, i, V = 1, 1, V0
        while i < self.maxiter and eps > self.tol:
            V1 = self.Vupdate(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < 0:
                nonmon = V1-V<0
                print("Failure of monotonicity at:", len(self.kgrid[nonmon]), "points")
                print("Average magnitude of failure:", np.mean(np.abs(V1[nonmon] - V[nonmon])))
            i = i+1
            V = V1
            print(eps)
        if i < self.maxiter:
            print("Policy function converged in:", i, "iterations")
        else:
            print("Policy function did not converge. Difference:", eps)
        return V

"""
Define parameters and solve for value function
"""

rho, delta, alpha = 0.1, 0.075, 1/3
kstar, A, B = 10, 1, 5
bnd, N = [1,80], 1000
tol, maxiter, Dt = 10**-6, 100, 10**-6

T = NCG(rho=rho,delta=delta,alpha=alpha,sigbar=0.0,kstar=kstar,A=A,B=B,bnd=bnd,N=N,maxiter=maxiter,tol=tol,Dt=Dt)
X = NCG(rho=rho,delta=delta,alpha=alpha,sigbar=0.2,kstar=kstar,A=A,B=B,bnd=bnd,N=N,maxiter=maxiter,tol=tol,Dt=Dt)
#T = classes_ex.NCG_zero_dt(rho=rho,delta=delta,alpha=alpha,sigbar=0.0,kstar=kstar,A=A,B=B,bnd=bnd,N=N,maxiter=maxiter,tol=tol)
#X = classes_ex.NCG_zero_dt(rho=rho,delta=delta,alpha=alpha,sigbar=0.2,kstar=kstar,A=A,B=B,bnd=bnd,N=N,maxiter=maxiter,tol=tol)

tic = time.time()
VX0 = X.Vupdate(X.f(X.kgrid)-X.delta*X.kgrid)
VX = X.solveV(VX0)
toc = time.time()
print("Time taken", toc-tic)

tic = time.time()
VT0 = T.Vupdate(T.f(T.kgrid)-T.delta*T.kgrid)
VT = T.solveV(VX)
toc = time.time()
print(toc-tic)

"""
Plot both value functions and associated policy functions.
"""

fig, ax = plt.subplots()
ax.plot(T.kgrid, VT,'b-',label="$\sigma = {0}$".format(T.sigbar),linewidth=1)
ax.plot(T.kgrid, VX,'c-',label="$\sigma = {0}$".format(X.sigbar),linewidth=1)
plt.xlabel('Capital $k$')
ax.legend(loc='upper left')
destin = '../../figures/NCG.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(T.kgrid, T.polupdate(VT),'b-',label="$\sigma = {0}$".format(T.sigbar),linewidth=1)
ax.plot(T.kgrid, X.polupdate(VX),'c-',label="$\sigma = {0}$".format(X.sigbar),linewidth=1)
plt.title('Consumption function')
plt.xlabel('Capital $k$')
plt.xlim(T.kgrid[0],T.kgrid[-1]*3/4)
plt.ylim(-1,20)
ax.legend(loc='upper left')
destin = '../../figures/NCGpol.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

Tdrift = T.f(T.kgrid) - T.polupdate(VT) - T.delta*T.kgrid
Xdrift = X.f(X.kgrid) - X.polupdate(VX) - X.delta*X.kgrid

fig, ax = plt.subplots()
ax.plot(T.kgrid, Tdrift,'b-',label="$\sigma = {0}$".format(T.sigbar),linewidth=1)
ax.plot(T.kgrid, Xdrift,'c-',label="$\sigma = {0}$".format(X.sigbar),linewidth=1)
plt.title('Drift: $f(k) - c - \delta k$')
plt.xlabel('Capital $k$')
plt.xlim(T.kgrid[0],T.kgrid[-1]*3/4)
plt.ylim(-5,5)
ax.legend(loc='upper right')
destin = '../../figures/NCGdrift.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(T.kgrid, T.sigma,'b-',label="$\sigma = {0}$".format(T.sigbar),linewidth=1)
ax.plot(T.kgrid, X.sigma,'c-',label="$\sigma = {0}$".format(X.sigbar),linewidth=1)
plt.title('Drift: $f(k) - c - \delta k$')
plt.xlabel('Capital $k$')
ax.legend(loc='upper right')
plt.show()
