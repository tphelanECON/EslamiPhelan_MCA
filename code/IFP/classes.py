"""
Original version: March 2022.

This version: March 2023.

Five classes: IFP_2D, GIFP_2D, IFP_3D, GIFP_3D, DuraCons.

IFP_2D is 2-dimensional IFP, GIFP_2D is generalized analogue.
IFP_3D is 3-dimensional IFP, GIFP_3D is generalized analogue.
DuraCons is durable consumption problem.

Following uses indexing='ij'. If xx, yy = np.meshgrid(xgrid, ygrid) then xx is
constant across columns and varies across rows.

Since numpy reshapes by reading across rows then columns
(i.e. np.array([[1,2],[3,4]]).reshape(-1) = np.array([1,2,3,4])), row number
i*(self.N[1]-1) + j corresponds to the transition probabilities
for the (i,j) point, and similarly for higher dimensions.
"""

import numpy as np
import time, scipy, scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class IFP_2D(object):
    def __init__(self,rho=1/0.95-1, r=0.03, gamma=2.,theta=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2,bnd=[[0,50],[-0.6,0.6]],N=(200,10),
    tol=10**-4, maxiter=200, kappa = 3):
        self.rho, self.gamma = rho, gamma
        self.r, self.theta, self.sigma = r, theta, sigma
        self.tol, self.maxiter, self.kappa = tol, maxiter, kappa
        self.N, self.M = N, (N[0]-1)*(N[1]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj= self.mesh([0,0])
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2) #vanish on boundaries
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + np.exp(self.xx[1])
        self.Dt = ((self.kappa-1)*self.c0/self.Delta[0] + (self.sigsig**2 \
        + self.Delta[1]*np.abs(self.theta*self.xx[1]))/self.Delta[1]**2)**(-1)
        self.cmax = self.kappa*self.c0
        self.V0 = self.V(self.c0)

    def p_func(self,ind,c):
        ii,jj = ind
        p_func, dum = {}, 0*self.xx[0][ii,jj]
        x = (self.xx[0][ii,jj],self.xx[1][ii,jj])
        dt, c, sig = self.Dt[ii,jj], c[ii,jj], self.sigsig[ii,jj]
        d = [dt/self.Delta[i]**2 for i in range(2)]
        p_func[(1,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+np.exp(x[1])-c,0)
        p_func[(-1,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+np.exp(x[1])-c),0)
        p_func[(0,1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(self.theta*(-x[1]),0))
        p_func[(0,-1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(-self.theta*(-x[1]),0))
        return p_func

    def P_tran(self,c):
        ii, jj = self.mesh([0,0])
        row = ii*(self.N[1]-1) + jj
        diag = 1 - sum(self.p_func((ii,jj),c).values())
        P = self.P_func(row,row,diag)
        for key in self.trans_keys:
            ii, jj = self.mesh(key)
            row = ii*(self.N[1]-1) + jj
            column = (ii+key[0])*(self.N[1]-1) + jj + key[1]
            P = P + self.P_func(row,column,self.p_func((ii,jj),c)[key])
        return P

    def V(self,c):
        b = self.Dt*c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = sp.eye(self.M) - sp.diags(D)*self.P_tran(c)
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1))

    def solve_PFI(self):
        V, i, eps = self.V0, 1, 1
        while i < 15 and eps > self.tol:
            tic = time.time()
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            toc = time.time()
            print("Time for last iteration:", toc-tic)
            print("Difference in iterations:", eps, "Iterations:", i)
            V, i = V1, i+1
        print("Difference in PFI:", eps, "Iterations:", i)
        return V

    def MVFI(self,c,V,M):
        b = (self.Dt*c**(1-self.gamma)/(1-self.gamma)).reshape((self.M,))
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        V, P = V.reshape((self.M,)), sp.diags(D)*self.P_tran(c)
        for i in range(M+1):
            V = b + P*V
        return V.reshape((self.N[0]-1,self.N[1]-1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 1, 1
        while i < self.maxiter and eps > self.tol:
            V1 = self.MVFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            V, i = V1, i+1
        print("Difference in iterates for MPFI", M, ":", eps, "Iterations:", i)
        print("Convergence?", i < self.maxiter)
        print("Consumption inequality holds?", np.max(self.polupdate(V) < self.cmax))
        return V

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1))
        Vbig[1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*self.Dt) \
        *(np.maximum(self.c0 - c, 0)*VF1 - np.maximum(-(self.c0 - c), 0)*VB1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum((np.exp(-self.rho*self.Dt)*VF1)**(-1/self.gamma), self.c0)
            chigh = np.maximum((np.exp(-self.rho*self.Dt)*VB1)**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax[VF1<=0], self.cmax[VB1<=0]
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:] = np.minimum(C[0,:], self.c0[0,:])
        C[-1,:] = np.maximum(C[-1,:], self.c0[-1,:])
        return C

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), indexing='ij')

"""
2D IFP with T = limit of (beta*P-I)/Deltat as Deltat rightarrow 0.
"""

class GIFP_2D(object):
    def __init__(self,rho=1/0.95-1, r=0.03, gamma=2.,theta=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2,bnd=[[0,50],[-0.6,0.6]],N=(1000,12),
    tol=10**-4,maxiter=200):
        self.rho, self.gamma = rho, gamma
        self.r, self.theta, self.sigma = r, theta, sigma
        self.tol, self.maxiter = tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(2)]
        self.ii, self.jj= self.mesh([0,0])
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2) #vanish on boundaries
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + np.exp(self.xx[1])
        self.cmax = 2*(self.r*self.bnd[0][1] + np.exp(self.bnd[1][1])+10)
        self.V0 = self.V(self.c0)

    def tran_func(self,ind,c):
        ii,jj = ind
        tran_func, dum = {}, 0*self.xx[0][ii,jj]
        x = (self.xx[0][ii,jj],self.xx[1][ii,jj])
        c, d = c[ii,jj], [1/self.Delta[i]**2 for i in range(2)]
        sig = self.sigsig[ii,jj]
        tran_func[(1,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+np.exp(x[1])-c,0)
        tran_func[(-1,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+np.exp(x[1])-c),0)
        tran_func[(0,1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(self.theta*(-x[1]),0))
        tran_func[(0,-1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(-self.theta*(-x[1]),0))
        return tran_func

    def norm_func(self,ind,c):
        norm = self.rho + sum(self.tran_func(ind,c).values())
        return {key:self.tran_func(ind,c)[key]/norm for key in self.trans_keys}

    def T_tran(self,c):
        row = self.ii*(self.N[1]-1) + self.jj
        T = self.T_func(row, row, -1 + 0*row)
        for key in self.trans_keys:
            ii, jj = self.mesh(key)
            row = ii*(self.N[1]-1) + jj
            column = (ii+key[0])*(self.N[1]-1) + jj + key[1]
            T = T + self.T_func(row, column, self.norm_func((ii,jj),c)[key])
        return T

    def V(self,c):
        norm = self.rho + sum(self.tran_func(self.mesh([0,0]),c).values())
        b = c**(1-self.gamma)/(1-self.gamma)
        return sp.linalg.spsolve(-self.T_tran(c), (b/norm).reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1))

    def solve_PFI(self):
        V, i, eps = self.V0, 1, 1
        while i < 15 and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            V, i = V1, i+1
        print("Difference in PFI:", eps, "Iterations:", i)
        return V

    def H(self,c):
        H = sp.coo_matrix((self.M,self.M))
        for key in self.trans_keys:
            ii, jj = self.mesh(key)
            row = ii*(self.N[1]-1) + jj
            column = (ii+key[0])*(self.N[1]-1) + jj + key[1]
            H = H + self.T_func(row, column, self.norm_func((ii,jj),c)[key])
        return H

    def MVFI(self,c,V,M):
        norm = self.rho + sum(self.tran_func(self.mesh([0,0]),c).values())
        b = c**(1-self.gamma)/(1-self.gamma)
        H, V = self.H(c), V.reshape((self.M,))
        for i in range(M+1):
            V = (b/norm).reshape((self.M,)) + H*V
        return V.reshape((self.N[0]-1,self.N[1]-1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 1, 1
        while i < self.maxiter and eps > self.tol:
            V1 = self.MVFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            V, i = V1, i+1
        print("Difference in iterates for MPFI", M,":", eps, "Iterations:", i)
        print("Convergence:", i < self.maxiter)
        return V

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1))
        Vbig[1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum(VF1**(-1/self.gamma), self.c0)
            chigh = np.maximum(VB1**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax, self.cmax
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) \
        + np.maximum(self.c0-c,0)*VF1 - np.maximum(-(self.c0-c),0)*VB1
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:] = np.minimum(C[0,:], self.c0[0,:])
        C[-1,:] = np.maximum(C[-1,:], self.c0[-1,:])
        return C

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), indexing='ij')

"""
Three-dimensional income-fluctuation problem. States: assets, log income 1, and log income 2
"""

class IFP_3D(object):
    def __init__(self,rho=1/0.95-1,r=0.03,gamma=2.,theta=[-np.log(0.95),-np.log(0.95)],
    sigma=[np.sqrt(-2*np.log(0.95))*0.2,np.sqrt(-2*np.log(0.95))*0.2],
    bnd=[[0,50],[-0.6,0.6],[-0.6,0.6]],N=(40,20,20),tol=10**-4,maxiter=200,
    mono_tol = 10**(-6),kappa = 3):
        self.rho, self.gamma = rho, gamma
        self.r, self.theta, self.sigma = r, theta, sigma
        self.tol, self.mono_tol, self.maxiter, self.kappa = tol,mono_tol, maxiter, kappa
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.ii, self.jj, self.kk = self.mesh([0,0,0])
        self.sigsig1 = self.sigma[0]*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.sigsig2 = self.sigma[1]*(self.kk > 0)*(self.kk < self.N[2] - 2)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.c0 = self.r*self.xx[0] + np.exp(self.xx[1]+self.xx[2])
        self.Dt = ((self.kappa-1)*self.c0/self.Delta[0] \
        + (self.sigsig1**2 + self.Delta[1]*np.abs(self.theta[0]*self.xx[1]))/self.Delta[1]**2 \
        + (self.sigsig2**2 + self.Delta[2]*np.abs(self.theta[1]*self.xx[2]))/self.Delta[2]**2)**(-1)
        self.cmax = self.kappa*self.c0
        self.V0 = self.V(self.c0)

    def p_func(self,ind,c):
        ii,jj,kk = ind
        p_func, dum = {}, 0*self.xx[0][ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        dt,c = self.Dt[ii,jj,kk], c[ii,jj,kk]
        sig1, sig2 = self.sigsig1[ii,jj,kk], self.sigsig2[ii,jj,kk]
        d = [dt/self.Delta[i]**2 for i in range(3)]
        p_func[(1,0,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+np.exp(x[1]+x[2])-c,0)
        p_func[(-1,0,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+np.exp(x[1]+x[2])-c),0)
        p_func[(0,1,0)] = d[1]*(sig1**2/2 + self.Delta[1]*np.maximum(self.theta[0]*(-x[1]),0))
        p_func[(0,-1,0)] = d[1]*(sig1**2/2 + self.Delta[1]*np.maximum(-self.theta[0]*(-x[1]),0))
        p_func[(0,0,1)] = d[2]*(sig2**2/2 + self.Delta[2]*np.maximum(self.theta[1]*(-x[2]),0))
        p_func[(0,0,-1)] = d[2]*(sig2**2/2 + self.Delta[2]*np.maximum(-self.theta[1]*(-x[2]),0))
        return p_func

    def P_tran(self,c):
        ii, jj, kk = self.mesh([0,0,0])
        row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
        diag = 1 - sum(self.p_func((ii,jj,kk),c).values())
        P = self.P_func(row,row,diag)
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*((self.N[1]-1)*(self.N[2]-1)) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            P = P + self.P_func(row,column,self.p_func((ii,jj,kk),c)[key])
        return P

    def V(self,c):
        b = self.Dt*c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = sp.eye(self.M) - sp.diags(D)*self.P_tran(c)
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def solve_PFI(self):
        V, i, eps = self.V0, 0, 1
        while i < 15 and eps > self.tol:
            tic = time.time()
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
            toc = time.time()
            print("Time for last iteration:", toc-tic)
            print("Difference in iterations:", eps, "Iterations:", i)
        print("Difference in PFI:", eps, "Iterations:", i)
        print("Convergence?", i < self.maxiter)
        print("Consumption inequality holds?", np.max(self.polupdate(V) < self.cmax))
        return V

    def MVFI(self,c,V,M):
        b = (self.Dt*c**(1-self.gamma)/(1-self.gamma)).reshape((self.M,))
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        V, P = V.reshape((self.M,)), sp.diags(D)*self.P_tran(c)
        for i in range(M+1):
            V = b + P*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 0, 1
        while i < self.maxiter and eps > self.tol:
            V1 = self.MVFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
        print("Difference in iterates for MPFI", M,":", eps, "Iterations:", i)
        print("Convergence?", i < self.maxiter)
        print("Consumption inequality holds?", np.max(self.polupdate(V) < self.cmax))
        return V

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1, self.N[2]+1))
        Vbig[1:-1,1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1,1:-1]-Vbig[1:-1,1:-1,1:-1])/self.Delta[0]
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*self.Dt) \
        *(np.maximum(self.c0 - c, 0)*VF1 - np.maximum(-(self.c0 - c), 0)*VB1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum((np.exp(-self.rho*self.Dt)*VF1)**(-1/self.gamma), self.c0)
            chigh = np.maximum((np.exp(-self.rho*self.Dt)*VB1)**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax[VF1<=0], self.cmax[VB1<=0]
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1,self.N[2]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:,:] = np.minimum(C[0,:,:], self.c0[0,:,:])
        C[-1,:,:] = np.maximum(C[-1,:,:], self.c0[-1,:,:])
        return C

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')

"""
3D IFP with zero timestep.
"""

class GIFP_3D(object):
    def __init__(self,rho=1/0.95-1,r=0.03,gamma=2.,theta=[-np.log(0.95),-np.log(0.95)],
    sigma=[np.sqrt(-2*np.log(0.95))*0.2,np.sqrt(-2*np.log(0.95))*0.2],
    bnd=[[0,50],[-0.6,0.6],[-0.6,0.6]],N=(40,20,20),tol=10**-4,maxiter=200,mono_tol = 10**(-6)):
        self.rho, self.gamma = rho, gamma
        self.r, self.theta, self.sigma = r, theta, sigma
        self.tol, self.mono_tol, self.maxiter = tol, mono_tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.ii, self.jj, self.kk = self.mesh([0,0,0])
        self.sigsig1 = self.sigma[0]*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.sigsig2 = self.sigma[1]*(self.kk > 0)*(self.kk < self.N[2] - 2)
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.c0 = self.r*self.xx[0] + np.exp(self.xx[1]+self.xx[2])
        self.cmax = 1.5*(self.r*self.bnd[0][1] + np.exp(self.bnd[1][1]+self.bnd[2][1])+1)
        self.V0 = self.V(self.c0)

    def tran_func(self,ind,c):
        ii,jj,kk = ind
        tran_func, dum = {}, 0*self.xx[0][ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        c, d = c[ii,jj,kk], [1/self.Delta[i]**2 for i in range(3)]
        sig1, sig2 = self.sigsig1[ii,jj,kk], self.sigsig2[ii,jj,kk]
        tran_func[(1,0,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+np.exp(x[1]+x[2])-c,0)
        tran_func[(-1,0,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+np.exp(x[1]+x[2])-c),0)
        tran_func[(0,1,0)] = d[1]*(sig1**2/2 + self.Delta[1]*np.maximum(self.theta[0]*(-x[1]),0))
        tran_func[(0,-1,0)] = d[1]*(sig1**2/2 + self.Delta[1]*np.maximum(-self.theta[0]*(-x[1]),0))
        tran_func[(0,0,1)] = d[2]*(sig2**2/2 + self.Delta[2]*np.maximum(self.theta[1]*(-x[2]),0))
        tran_func[(0,0,-1)] = d[2]*(sig2**2/2 + self.Delta[2]*np.maximum(-self.theta[1]*(-x[2]),0))
        return tran_func

    def norm_func(self,ind,c):
        norm = self.rho + sum(self.tran_func(ind,c).values())
        return {key:self.tran_func(ind,c)[key]/norm for key in self.trans_keys}

    def T_tran(self,c):
        row = self.ii*((self.N[1]-1)*(self.N[2]-1)) + self.jj*(self.N[2]-1) + self.kk
        T = self.T_func(row, row, -1 + 0*row)
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*(self.N[1]-1)*(self.N[2]-1) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*(self.N[1]-1)*(self.N[2]-1) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            T = T + self.T_func(row, column, self.norm_func((ii,jj,kk),c)[key])
        return T

    def V(self,c):
        norm = self.rho + sum(self.tran_func(self.mesh([0,0,0]),c).values())
        b = c**(1-self.gamma)/(1-self.gamma)
        return sp.linalg.spsolve(-self.T_tran(c), (b/norm).reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def solve_PFI(self):
        V, i, eps = self.V0, 0, 1
        while i < 30 and eps > self.tol:
            tic = time.time()
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
            toc = time.time()
            print("Time for last iteration:", toc-tic)
            print("Difference in iterations:", eps, "Iterations:", i)
        print("Difference in PFI:", eps, "Iterations:", i)
        print("Convergence?", i < self.maxiter)
        return V

    def H(self,c):
        H = sp.coo_matrix((self.M,self.M))
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*(self.N[1]-1)*(self.N[2]-1) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*(self.N[1]-1)*(self.N[2]-1) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            H = H + self.T_func(row, column, self.norm_func((ii,jj,kk),c)[key])
        return H

    def MVFI(self,c,V,M):
        norm = self.rho + sum(self.tran_func(self.mesh([0,0,0]),c).values())
        b = c**(1-self.gamma)/(1-self.gamma)
        H, V = self.H(c), V.reshape((self.M,))
        for i in range(M+1):
            V = (b/norm).reshape((self.M,)) + H*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 0, 1
        while i < self.maxiter and eps > self.tol:
            V1 = self.MVFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
        print("Difference in iterates for MPFI", M,":", eps, "Iterations:", i)
        print("Convergence?", i < self.maxiter)
        return V

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1, self.N[2]+1))
        Vbig[1:-1,1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1,1:-1]-Vbig[1:-1,1:-1,1:-1])/self.Delta[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum(VF1**(-1/self.gamma), self.c0)
            chigh = np.maximum(VB1**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax, self.cmax
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) \
        + np.maximum(self.c0-c,0)*VF1 - np.maximum(-(self.c0-c),0)*VB1
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1,self.N[2]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:,:] = np.minimum(C[0,:,:], self.c0[0,:,:])
        C[-1,:,:] = np.maximum(C[-1,:,:], self.c0[-1,:,:])
        return C

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')

"""
Durable consumption IFP. States: wealth, income and durable good.
Asset grid NOT determined by bnd and N[0]. Exogenous K indicates no. changes in
asset grid with increase in durable good. Must start with V s.t. B(V) geq 0.
"""

class DuraCons(object):
    def __init__(self, rho=1/0.95-1,r=0.03, eta=1/0.77-1, iota=0.1, theta=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, pbar=1, bnd=[[0,100],[-0.6,0.6],[0,10]],
    N=(60,20,10), K=2, lambar=100, tol=10**-5, mono_tol = 10**(-6),maxiter=100):
        self.r, self.eta, self.iota, self.rho = r, eta, iota, rho
        self.theta, self.sigma, self.pbar = theta, sigma, pbar
        self.tol, self.mono_tol, self.maxiter = tol, mono_tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.lambar, self.K = lambar, K
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.Delta[0] = self.pbar*self.Delta[2]/self.K
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][0]+(self.N[i]-1)*self.Delta[i],self.N[i]-1) for i in range(3)]
        self.ii, self.jj, self.kk = self.mesh([0,0,0])
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(-self.K,0,1)]
        self.c0 = self.r*self.xx[0] + np.exp(self.xx[1])
        self.cmax = 2.5*(self.r*self.bnd[0][1] + np.exp(self.bnd[1][1]))
        self.cmin = 0.5*(self.r*self.bnd[0][0] + np.exp(self.bnd[1][0]))
        self.V0 = self.V((self.c0,0*self.c0))

    def u(self,pol):
        return np.log(pol[0]) + self.eta*np.log(self.xx[2]+self.iota)

    def tran_func(self,ind,pol):
        (ii,jj,kk), tran_func = ind, {}
        c, lam = pol[0][ii,jj,kk], pol[1][ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        sig = self.sigsig[ii,jj,kk]
        tran_func[(1,0,0)] = (1/self.Delta[0])*np.maximum(self.r*x[0]+np.exp(x[1])-c,0)
        tran_func[(-1,0,0)] = (1/self.Delta[0])*np.maximum(-(self.r*x[0]+np.exp(x[1])-c),0)
        tran_func[(0,1,0)] = (1/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(self.theta*(-x[1]),0))
        tran_func[(0,-1,0)] = (1/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(-self.theta*(-x[1]),0))
        tran_func[(-self.K,0,1)] = lam
        return tran_func

    def norm_func(self,ind,pol):
        norm = self.rho + sum(self.tran_func(ind,pol).values())
        return {key:self.tran_func(ind,pol)[key]/norm for key in self.trans_keys}

    def T_tran(self,pol):
        row = self.ii*((self.N[1]-1)*(self.N[2]-1)) + self.jj*(self.N[2]-1) + self.kk
        T = self.T_func(row, row, -1 + 0*row)
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*(self.N[1]-1)*(self.N[2]-1) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*(self.N[1]-1)*(self.N[2]-1) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            T = T + self.T_func(row, column, self.norm_func((ii,jj,kk),pol)[key])
        return T

    def H(self,pol):
        H = sp.coo_matrix((self.M,self.M))
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*(self.N[1]-1)*(self.N[2]-1) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*(self.N[1]-1)*(self.N[2]-1) + (jj+key[1])*(self.N[2]-1) + kk+key[2]
            H = H + self.T_func(row, column, self.norm_func((ii,jj,kk),pol)[key])
        return H

    def V(self,pol):
        b = self.u(pol)/(self.rho+sum(self.tran_func(self.mesh([0,0,0]),pol).values()))
        return sp.linalg.spsolve(-self.T_tran(pol), b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    #solve using policy function iteration.
    def solve_PFI(self):
        V, i, eps, eps2 = self.V0, 0, 1, 1
        while i < 80 and eps2 > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.sum(np.abs(V1-V)/self.M)
            eps2 = np.max(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
            print("Differences in PFI iterations:", eps2, "Iterations:", i)
        return V

    def MPFI(self,pol,V,M):
        b = self.u(pol)/(self.rho+sum(self.tran_func(self.mesh([0,0,0]),pol).values()))
        H, V = self.H(pol), V.reshape((self.M,))
        for i in range(M+1):
            V = b.reshape((self.M,)) + H*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def solve_MPFI(self,M):
        V, i, eps, eps2 = self.V0, 0, 1, 1
        while i < self.maxiter and eps2 > self.tol:
            V1 = self.MPFI(self.polupdate(V),V,M)
            eps = np.sum(np.abs(V1-V)/self.M)
            eps2 = np.max(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
        print("MPFI with", M, "relaxations,", self.N, "gridpoints:", "Differences:", (eps2), "Iterations:", i)
        return V

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1, self.N[2]+1))
        Vbig[1:-1,1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1,1:-1]-Vbig[1:-1,1:-1,1:-1])/self.Delta[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.maximum(np.minimum(VF1**(-1), self.c0),self.cmin)
            chigh = np.minimum(np.maximum(VB1**(-1), self.c0), self.cmax)
        clow[VF1<=0], chigh[VB1<=0] = self.c0[VF1<=0], self.c0[VB1<=0]
        obj = lambda c: np.log(c) + np.maximum(self.c0-c,0)*VF1 - np.maximum(-(self.c0-c),0)*VB1
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1,self.N[2]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        ind_lam = np.roll(np.roll(V,self.K,axis=0),-1,axis=2) > V
        ind_lam[:self.K+1,:,:], ind_lam[:,:,-1] = 0, 0
        return np.minimum(self.cmax,C), self.lambar*ind_lam

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')
