"""
Classes for linear-quadratic problems. Independent noise.

LQ_2D: 2D problem; two controls; parameters chosen s.t. drift negative
LQ_3D: 3D problem; one control; parameters chosen s.t. drift negative
LQ_3D_SD: 3D problem with state-dependent timestep.
LQ_3D_GEN: generalized normalized policy function.

All use 'ij' indexing for meshgrid.
"""

import numpy as np
from numba import jit
import scipy
import scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools
import timeit
import time
import math

class LQ_2D(object):
    def __init__(self,Q,R,A,B,sigma,rho,N,Deltat,x_bnd=[[0,2],[0,2]],tol=10**-6):
        self.Q, self.R, self.A, self.B, self.x_bnd = Q, R, A, B, x_bnd
        self.sigma, self.rho = sigma, rho
        self.N, self.M, self.Deltat, self.tol = N, (N[0]-1)*(N[1]-1), Deltat, tol
        self.Delta = (self.x_bnd[0][1]-self.x_bnd[0][0])/self.N[0], (self.x_bnd[1][1]-self.x_bnd[1][0])/self.N[1]
        self.Abar = self.A-self.rho*np.eye(2)/2
        self.P = scipy.linalg.solve_continuous_are(self.Abar,self.B,self.Q,self.R)
        self.con = (np.mat(self.sigma)*np.mat(self.sigma).T*np.mat(self.P)).trace()/(2*self.rho)
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.check_drift = self.A - np.mat(B)*np.linalg.inv(self.R)*np.mat(B.T)*np.mat(self.P)
        self.CF = self.CF_fun(self.grid(1))

    def CF_fun(self,x):
        return -self.mat_quad(self.P,x)/2 - np.array(self.con)

    def mu(self,x,u):
        return self.mat_array(self.A,x)[0] + self.mat_array(self.B,u)[0], \
        self.mat_array(self.A,x)[1] + self.mat_array(self.B,u)[1]

    def p_func(self,ind,u):
        ii,jj = ind
        p_func, dum = {}, 0*self.grid(1)[0][ii,jj]
        x, u = (self.grid(1)[0][ii,jj],self.grid(1)[1][ii,jj]), (u[0][ii,jj],u[1][ii,jj])
        d = self.Deltat/self.Delta[0]**2, self.Deltat/self.Delta[1]**2
        p_func[(1,0)] = d[0]*(self.sigma[0]**2/2 + self.Delta[0]*np.maximum(self.mu(x,u)[0],0))
        p_func[(-1,0)] = d[0]*(self.sigma[0]**2/2 + self.Delta[0]*np.maximum(-self.mu(x,u)[0],0))
        p_func[(0,1)] = d[1]*(self.sigma[1]**2/2 + self.Delta[1]*np.maximum(self.mu(x,u)[1],0))
        p_func[(0,-1)] = d[1]*(self.sigma[1]**2/2 + self.Delta[1]*np.maximum(-self.mu(x,u)[1],0))
        return p_func

    def P_tran(self,u):
        ii, jj = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        P = self.P_func(ii*(self.N[1]-1) + jj, ii*(self.N[1]-1) + jj, 1 - sum(self.p_func((ii,jj),u).values()))
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]-1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]-1-max(key[1],0)),indexing='ij')
            row = ii*(self.N[1]-1) + jj
            column = (ii+key[0])*(self.N[1]-1) + jj + key[1]
            P = P + self.P_func(row,column,self.p_func((ii,jj),u)[key])
        return P

    def polupdate(self,V):
        Vbig = self.CF_fun(self.grid(0))
        Vbig[1:-1,1:-1] = V
        VB0 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VB1 = (Vbig[1:-1,1:-1]-Vbig[1:-1,:-2])/self.Delta[1]
        obj = lambda u: - np.exp(self.rho*self.Deltat)*self.mat_quad(self.R,u)/2 \
        - np.maximum(-self.mu((self.grid(1)),u)[0],0)*VB0 \
        - np.maximum(-self.mu((self.grid(1)),u)[1],0)*VB1
        F = lambda u: -obj(u).reshape((self.M,))
        #nterior:
        xx = self.grid(1)
        Ax = (self.mat_array(self.A,xx)[0], self.mat_array(self.A,xx)[1])
        C = (np.exp(-self.rho*self.Deltat)*VB0, np.exp(-self.rho*self.Deltat)*VB1)
        int_opt = np.minimum(self.mat_array(np.linalg.inv(self.R),C),-np.array(Ax))
        #now edges:
        E = ((np.exp(-self.rho*self.Deltat)*VB0 + self.R[0,1]*Ax[0])/self.R[0,0], \
        (np.exp(-self.rho*self.Deltat)*VB1 + self.R[0,1]*Ax[1])/self.R[1,1])
        edge = [(-Ax[0], np.minimum(E[1],0)), (np.minimum(E[0],0), -Ax[1])]
        Z = np.vstack([F(int_opt),F(edge[0]),F(edge[1])])
        I = np.argmin(Z, axis=0).reshape((self.N[0]-1,self.N[1]-1))
        return (I==0)*int_opt + (I==1)*edge[0] + (I==2)*edge[1]

    def V(self,u):
        b = -self.Deltat*(self.mat_quad(self.Q,self.grid(1))+self.mat_quad(self.R,u))/2 \
        + np.exp(-self.rho*self.Deltat)*self.BND(u)
        B = sp.eye(self.M) - np.exp(-self.rho*self.Deltat)*self.P_tran(u)
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[1]-1,self.N[0]-1))

    def V_MPFI(self,V,u,M):
        b = -self.Deltat*(self.mat_quad(self.Q,self.grid(1))+self.mat_quad(self.R,u))/2 \
        + np.exp(-self.rho*self.Deltat)*self.BND(u)
        V, b, i = V.reshape((self.M,)), b.reshape((self.M,)), 1
        for i in range(M+1):
            V = b + np.exp(-self.rho*self.Deltat)*self.P_tran(u)*V
        return V.reshape((self.N[1]-1,self.N[0]-1))

    def solve_PFI(self):
        u = (0*self.grid(1)[0],0*self.grid(1)[1])
        V,eps,i = self.V(u),1,1
        while i < 20 and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
            print("Error in consumer problem:",eps)
        return V

    def solve_MPFI(self,M):
        u = (0*self.grid(1)[0],0*self.grid(1)[1])
        V, eps, i = self.V(u), 1, 1
        while i < 20 and eps > self.tol:
            V1 = self.V_PFI(V,self.polupdate(V),M)
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
            print("Error in consumer problem:",eps)
        return V

    def BND(self,u):
        CF = self.CF_fun(self.grid(0))
        BND = np.zeros((self.N[0]-1,self.N[1]-1))
        p = self.p_func((0,np.arange(0,self.N[1]-1)),u)
        BND[0,:] = p[(-1,0)]*CF[0,1:-1]
        p = self.p_func((self.N[0]-2,np.arange(0,self.N[1]-1)),u)
        BND[-1,:] = p[(1,0)]*CF[-1,1:-1]
        p = self.p_func((np.arange(0,self.N[0]-1),0),u)
        BND[:,0] = BND[:,0] + p[(0,-1)]*CF[1:-1,0]
        p = self.p_func((np.arange(0,self.N[0]-1),self.N[1]-2),u)
        BND[:,-1] = BND[:,-1] + p[(0,1)]*CF[1:-1,-1]
        return BND

    def grid(self,m):
        x1 = np.linspace(self.x_bnd[0][0]+m*self.Delta[0],self.x_bnd[0][0]+(self.N[0]-m)*self.Delta[0],self.N[0]+1-2*m)
        x2 = np.linspace(self.x_bnd[1][0]+m*self.Delta[1],self.x_bnd[1][0]+(self.N[1]-m)*self.Delta[1],self.N[1]+1-2*m)
        return np.meshgrid(x1,x2,indexing='ij')

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def mat_quad(self,M,z):
        return M[0,0]*z[0]**2 + M[0,1]*z[1]*z[0] + M[1,0]*z[1]*z[0] + M[1,1]*z[1]**2

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1], M[1,0]*z[0] + M[1,1]*z[1]

"""
LQ control in 3D. Single control. Choose parameters s.t. drift non-positive.
"""

class LQ_3D(object):
    def __init__(self, rho=.1, Q=1*np.eye(3), R=np.eye(1), A=0.01*np.eye(3),
    B = np.array([.025,.025,.025]).reshape(3,1), sigma=0.4*np.eye(3), N=(40,30,30),
    Deltat=2*10**-2,bnd=[[0, 10],[0, 10],[0, 10]],tol=10**-6, maxiter=200):
        self.rho, self.Q, self.R, self.A, self.B, self.sigma = rho, Q, R, A, B, sigma
        self.Deltat, self.tol, self.maxiter = Deltat, tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i], self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        big_grid = [np.linspace(self.bnd[i][0], self.bnd[i][1],self.N[i]+1) for i in range(3)]
        self.big_xx = np.meshgrid(big_grid[0],big_grid[1],big_grid[2],indexing='ij')
        self.Abar = self.A-self.rho*np.eye(3)/2
        self.P = scipy.linalg.solve_continuous_are(self.Abar,self.B,self.Q,np.eye(1))
        self.con = (np.mat(self.sigma)*np.mat(self.sigma).T*np.mat(self.P)).trace()/(2*self.rho)
        self.CF = self.CF_fun(self.xx)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.drift = self.A - np.mat(self.B)*np.mat(self.B.T)*np.mat(self.P)

    def obj(self,x,u):
        return -self.mat_quad(self.Q,x)/2 - u**2/2

    def CF_fun(self,x):
        return -self.mat_quad(self.P,x)/2 - np.array(self.con)

    def p_func(self,ind,u):
        (ii,jj,kk) = ind
        p_func, u = {}, u[ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        drift = [self.mat_array(self.A,x)[i] + self.B[i]*u for i in range(3)]
        d = [self.Deltat/self.Delta[i]**2 for i in range(3)]
        p_func[(1,0,0)] = d[0]*(self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(drift[0],0))
        p_func[(-1,0,0)] = d[0]*(self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(-drift[0],0))
        p_func[(0,1,0)] = d[1]*(self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(drift[1],0))
        p_func[(0,-1,0)] = d[1]*(self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(-drift[1],0))
        p_func[(0,0,1)] = d[2]*(self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(drift[2],0))
        p_func[(0,0,-1)] = d[2]*(self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(-drift[2],0))
        return p_func

    def P_tran(self,u):
        ii, jj, kk = self.mesh([0,0,0])
        row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
        P = self.P_func(row, row, 1 - sum(self.p_func((ii,jj,kk),u).values()))
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*((self.N[1]-1)*(self.N[2]-1)) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            P = P+self.P_func(row,column,self.p_func((ii,jj,kk),u)[key])
        return P

    def V(self,u):
        b = self.Deltat*self.obj(self.xx,u) + np.exp(-self.rho*self.Deltat)*self.BND(u)
        B = sp.eye(self.M) - np.exp(-self.rho*self.Deltat)*self.P_tran(u)
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def polupdate(self,V):
        Vbig, Ax = self.CF_fun(self.big_xx), self.mat_array(self.A,self.xx)
        Vbig[1:-1,1:-1,1:-1] = V
        VB = [(Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,:-2,1:-1])/self.Delta[1],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,1:-1,:-2])/self.Delta[2]]
        u_int = np.exp(-self.rho*self.Deltat)*(self.B[0]*VB[0]+self.B[1]*VB[1]+self.B[2]*VB[2])
        m = np.minimum(-Ax[0]/self.B[0],-Ax[1]/self.B[1],-Ax[2]/self.B[2])
        return np.minimum(u_int,m)

    def solve_PFI(self):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.u)
        while i < 10 and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs((V1 - V)))
            V, i = V1, i+1
            if np.min(V1-V) < -10**(-6):
                print("Failure of monotonicity at:", len(V[V1-V<-10**(-6)]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-10**(-6)]))
        print("Difference between iterations:", eps, "Iterations:", i)
        return V

    def MPFI(self,u,V,M):
        b = (self.Deltat*self.obj(self.xx,u) + np.exp(-self.rho*self.Deltat)*self.BND(u)).reshape((self.M,))
        V, P = V.reshape((self.M,)), np.exp(-self.rho*self.Deltat)*self.P_tran(u)
        for i in range(M+1):
            V = b + P*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def Update_MPFI(self,V,M):
        return self.MPFI(self.polupdate(V),V,M)

    def solve_MPFI(self,M):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.u)
        while i < self.maxiter and eps > self.tol:
            V1 = self.Update_MPFI(V,M)
            eps = np.amax(np.abs(V1 - V))
            print("Difference between iterations:", eps, "Iterations:", i)
            V, i = V1, i+1
        return V

    def BND(self,u):
        CF = self.CF_fun(self.big_xx)
        BND = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        mesh12 = np.meshgrid(range(0,self.N[1]-1), range(0,self.N[2]-1), indexing='ij')
        mesh02 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[2]-1), indexing='ij')
        mesh01 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[1]-1), indexing='ij')
        p = self.p_func((0,mesh12[0],mesh12[1]),u)
        BND[0,:,:] = p[(-1,0,0)]*CF[0,1:-1,1:-1]
        p = self.p_func((self.N[0]-2,mesh12[0],mesh12[1]),u)
        BND[-1,:,:] = p[(1,0,0)]*CF[-1,1:-1,1:-1]
        p = self.p_func((mesh02[0],0,mesh02[1]),u)
        BND[:,0,:] = BND[:,0,:] + p[(0,-1,0)]*CF[1:-1,0,1:-1]
        p = self.p_func((mesh02[0],self.N[1]-2,mesh02[1]),u)
        BND[:,-1,:] = BND[:,-1,:] + p[(0,1,0)]*CF[1:-1,-1,1:-1]
        p = self.p_func((mesh01[0],mesh01[1],0),u)
        BND[:,:,0] = BND[:,:,0] + p[(0,0,-1)]*CF[1:-1,1:-1,0]
        p = self.p_func((mesh01[0],mesh01[1],self.N[2]-2),u)
        BND[:,:,-1] = BND[:,:,-1] + p[(0,0,1)]*CF[1:-1,1:-1,-1]
        return BND

    def mat_quad(self,M,z):
        return M[0,0]*z[0]*z[0] + M[0,1]*z[1]*z[0] + M[0,2]*z[2]*z[0] \
        + M[1,0]*z[0]*z[1] + M[1,1]*z[1]*z[1] + M[1,2]*z[2]*z[1] \
        + M[2,0]*z[0]*z[2] + M[2,1]*z[1]*z[2] + M[2,2]*z[2]*z[2]

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1] + M[0,2]*z[2], \
        M[1,0]*z[0] + M[1,1]*z[1] + M[1,2]*z[2], \
        M[2,0]*z[0] + M[2,1]*z[1] + M[2,2]*z[2]

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')

"""
State-dependent timestep. Define u lower as policy function one would pick if
finite-differences that appear in policy function are replaced with Q/rho.
"""

class LQ_3D_SD(object):
    def __init__(self, rho=.1, Q=1*np.eye(3), A=0.01*np.eye(3),
    B = np.array([.025,.025,.025]).reshape(3,1), sigma=0.4*np.eye(3), N=(40,40,40),
    bnd=[[0, 10],[0, 10],[0, 10]],tol=10**-6, maxiter=2000, kappa = 3):
        self.rho, self.Q, self.A, self.B, self.sigma = rho, Q, A, B, sigma
        self.tol, self.maxiter = tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i], self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        big_grid = [np.linspace(self.bnd[i][0], self.bnd[i][1],self.N[i]+1) for i in range(3)]
        self.big_xx = np.meshgrid(big_grid[0],big_grid[1],big_grid[2],indexing='ij')
        self.Abar = self.A-self.rho*np.eye(3)/2
        self.P = scipy.linalg.solve_continuous_are(self.Abar,self.B,self.Q,np.eye(1))
        self.con = (np.mat(self.sigma)*np.mat(self.sigma).T*np.mat(self.P)).trace()/(2*self.rho)
        self.CF = self.CF_fun(self.xx)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.drift = self.A - np.mat(self.B)*np.mat(self.B.T)*np.mat(self.P)
        u_mat = np.mat(self.B.T)*np.mat(self.P)
        self.u = - (u_mat[0,0]*self.xx[0] + u_mat[0,1]*self.xx[1] + u_mat[0,2]*self.xx[2])
        self.kappa = kappa
        self.u_low = self.kappa*self.u

    def obj(self,x,u):
        return -self.mat_quad(self.Q,x)/2 - u**2/2

    def CF_fun(self,x):
        return -self.mat_quad(self.P,x)/2 - np.array(self.con)

    def p_func(self,ind,u):
        (ii,jj,kk) = ind
        p_func, u = {}, u[ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        drift = [self.mat_array(self.A,x)[i] + self.B[i]*u for i in range(3)]
        d = [self.Deltat()[ii,jj,kk]/self.Delta[i]**2 for i in range(3)]
        p_func[(1,0,0)] = d[0]*(self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(drift[0],0))
        p_func[(-1,0,0)] = d[0]*(self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(-drift[0],0))
        p_func[(0,1,0)] = d[1]*(self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(drift[1],0))
        p_func[(0,-1,0)] = d[1]*(self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(-drift[1],0))
        p_func[(0,0,1)] = d[2]*(self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(drift[2],0))
        p_func[(0,0,-1)] = d[2]*(self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(-drift[2],0))
        return p_func

    def P_tran(self,u):
        ii, jj, kk = self.mesh([0,0,0])
        row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
        P = self.P_func(row, row, 1 - sum(self.p_func((ii,jj,kk),u).values()))
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*(self.N[1]-1)*(self.N[2]-1) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            P = P+self.P_func(row,column,self.p_func((ii,jj,kk),u)[key])
        return P

    def V(self,u):
        b = self.Deltat()*self.obj(self.xx,u) + np.exp(-self.rho*self.Deltat())*self.BND(u)
        D = np.exp(-self.rho*self.Deltat()).reshape((self.M,))
        B = sp.eye(self.M) - sp.diags(D)*self.P_tran(u)
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def polupdate(self,V):
        Vbig, Ax = self.CF_fun(self.big_xx), self.mat_array(self.A,self.xx)
        Vbig[1:-1,1:-1,1:-1] = V
        VB = [(Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,:-2,1:-1])/self.Delta[1],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,1:-1,:-2])/self.Delta[2]]
        u_int = np.exp(-self.rho*self.Deltat())*(self.B[0]*VB[0]+self.B[1]*VB[1]+self.B[2]*VB[2])
        m = np.minimum(-Ax[0]/self.B[0],-Ax[1]/self.B[1],-Ax[2]/self.B[2])
        return np.maximum(np.minimum(u_int,m),self.u_low)

    def solve_PFI(self):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.u)
        while i < 10 and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs((V1 - V)))
            if np.min(V1-V) < 0:
                print("Failure of monotonicity at:", len(V[V1-V<0]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<0]))
            print("Difference between iterations:", eps, "Iterations:", i)
            V, i = V1, i+1
        return V

    def MPFI(self,u,V,M):
        b = (self.Deltat()*self.obj(self.xx,u) + np.exp(-self.rho*self.Deltat())*self.BND(u)).reshape((self.M,))
        D = np.exp(-self.rho*self.Deltat()).reshape((self.M,))
        P = sp.diags(D)*self.P_tran(u)
        V = V.reshape((self.M,))
        for i in range(M+1):
            V = b + P*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def Update_MPFI(self,V,M):
        return self.MPFI(self.polupdate(V),V,M)

    def solve_MPFI(self,M):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.u)
        while i < self.maxiter and eps > self.tol:
            V1 = self.Update_MPFI(V,M)
            eps = np.amax(np.abs(V1 - V))
            print("Difference between iterations:", eps, "Iterations:", i)
            V, i = V1, i+1
        return V

    def BND(self,u):
        CF = self.CF_fun(self.big_xx)
        BND = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        mesh12 = np.meshgrid(range(0,self.N[1]-1), range(0,self.N[2]-1), indexing='ij')
        mesh02 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[2]-1), indexing='ij')
        mesh01 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[1]-1), indexing='ij')
        p = self.p_func((0,mesh12[0],mesh12[1]),u)
        BND[0,:,:] = p[(-1,0,0)]*CF[0,1:-1,1:-1]
        p = self.p_func((self.N[0]-2,mesh12[0],mesh12[1]),u)
        BND[-1,:,:] = p[(1,0,0)]*CF[-1,1:-1,1:-1]
        p = self.p_func((mesh02[0],0,mesh02[1]),u)
        BND[:,0,:] = BND[:,0,:] + p[(0,-1,0)]*CF[1:-1,0,1:-1]
        p = self.p_func((mesh02[0],self.N[1]-2,mesh02[1]),u)
        BND[:,-1,:] = BND[:,-1,:] + p[(0,1,0)]*CF[1:-1,-1,1:-1]
        p = self.p_func((mesh01[0],mesh01[1],0),u)
        BND[:,:,0] = BND[:,:,0] + p[(0,0,-1)]*CF[1:-1,1:-1,0]
        p = self.p_func((mesh01[0],mesh01[1],self.N[2]-2),u)
        BND[:,:,-1] = BND[:,:,-1] + p[(0,0,1)]*CF[1:-1,1:-1,-1]
        return BND

    #determine largest timestep s.t. probabilities remain in unit interval:
    def Deltat(self):
        return (self.sigma[0,0]**2/self.Delta[0]**2 \
        + self.sigma[1,1]**2/self.Delta[1]**2 + self.sigma[2,2]**2/self.Delta[2]**2 \
        + np.maximum(-(self.mat_array(self.A,self.xx)[0] + self.B[0]*self.u_low),0)/self.Delta[0] \
        + np.maximum(-(self.mat_array(self.A,self.xx)[1] + self.B[1]*self.u_low),0)/self.Delta[1] \
        + np.maximum(-(self.mat_array(self.A,self.xx)[2] + self.B[2]*self.u_low),0)/self.Delta[2])**(-1)

    def mat_quad(self,M,z):
        return M[0,0]*z[0]*z[0] + M[0,1]*z[1]*z[0] + M[0,2]*z[2]*z[0] \
        + M[1,0]*z[0]*z[1] + M[1,1]*z[1]*z[1] + M[1,2]*z[2]*z[1] \
        + M[2,0]*z[0]*z[2] + M[2,1]*z[1]*z[2] + M[2,2]*z[2]*z[2]

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1] + M[0,2]*z[2], \
        M[1,0]*z[0] + M[1,1]*z[1] + M[1,2]*z[2], \
        M[2,0]*z[0] + M[2,1]*z[1] + M[2,2]*z[2]

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')

"""
We now consider generalized modified policy function iteration.
"""

class LQ_3D_GEN(object):
    def __init__(self, rho=.1, Q=1*np.eye(3), A=0.01*np.eye(3),
    B = np.array([.025,.025,.025]).reshape(3,1), sigma=0.4*np.eye(3), N=(40,30,30),
    bnd=[[0, 10],[0, 10],[0, 10]],tol=10**-6, maxiter=2000):
        self.rho, self.Q, self.A, self.B, self.sigma = rho, Q, A, B, sigma
        self.tol, self.maxiter = tol, maxiter
        self.N, self.M = N, (N[0]-1)*(N[1]-1)*(N[2]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i], self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.ii, self.jj, self.kk = self.mesh([0,0,0])
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        big_grid = [np.linspace(self.bnd[i][0], self.bnd[i][1],self.N[i]+1) for i in range(3)]
        self.big_xx = np.meshgrid(big_grid[0],big_grid[1],big_grid[2],indexing='ij')
        self.Abar = self.A-self.rho*np.eye(3)/2
        self.P = scipy.linalg.solve_continuous_are(self.Abar,self.B,self.Q,np.eye(1))
        self.con = (np.mat(self.sigma)*np.mat(self.sigma).T*np.mat(self.P)).trace()/(2*self.rho)
        self.CF = self.CF_fun(self.xx)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.drift = self.A - np.mat(self.B)*np.mat(self.B.T)*np.mat(self.P)

    def obj(self,x,u):
        return -self.mat_quad(self.Q,x)/2 - u**2/2

    def CF_fun(self,x):
        return -self.mat_quad(self.P,x)/2 - np.array(self.con)

    def tran_func(self,ind,u):
        (ii,jj,kk) = ind
        tran_func, u = {}, u[ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        drift = [self.mat_array(self.A,x)[i] + self.B[i]*u for i in range(3)]
        tran_func[(1,0,0)] = (self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(drift[0],0))/self.Delta[0]**2
        tran_func[(-1,0,0)] = (self.sigma[0,0]**2/2 + self.Delta[0]*np.maximum(-drift[0],0))/self.Delta[0]**2
        tran_func[(0,1,0)] = (self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(drift[1],0))/self.Delta[1]**2
        tran_func[(0,-1,0)] = (self.sigma[1,1]**2/2 + self.Delta[1]*np.maximum(-drift[1],0))/self.Delta[1]**2
        tran_func[(0,0,1)] = (self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(drift[2],0))/self.Delta[2]**2
        tran_func[(0,0,-1)] = (self.sigma[2,2]**2/2 + self.Delta[2]*np.maximum(-drift[2],0))/self.Delta[2]**2
        return tran_func

    def norm_func(self,ind,u):
        norm = self.rho + sum(self.tran_func(ind,u).values())
        return {key:self.tran_func(ind,u)[key]/norm for key in self.trans_keys}

    def T_tran(self,u):
        row = self.ii*((self.N[1]-1)*(self.N[2]-1)) + self.jj*(self.N[2]-1) + self.kk
        T = self.T_func(row, row, -1 + 0*row)
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*((self.N[1]-1)*(self.N[2]-1)) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            T = T + self.T_func(row, column, self.norm_func((ii,jj,kk),u)[key])
        return T

    def H(self,u):
        H = sp.coo_matrix((self.M,self.M))
        for key in self.trans_keys:
            ii, jj, kk = self.mesh(key)
            row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
            column = (ii+key[0])*((self.N[1]-1)*(self.N[2]-1)) + (jj+key[1])*(self.N[2]-1) + kk + key[2]
            H = H + self.T_func(row, column, self.norm_func((ii,jj,kk),u)[key])
        return H

    def V(self,u):
        norm = self.rho + sum(self.tran_func(self.mesh([0,0,0]),u).values())
        b = self.obj(self.xx,u) + self.BND(u)
        return sp.linalg.spsolve(-self.T_tran(u), (b/norm).reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def polupdate(self,V):
        Vbig, Ax = self.CF_fun(self.big_xx), self.mat_array(self.A,self.xx)
        Vbig[1:-1,1:-1,1:-1] = V
        VB = [(Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,:-2,1:-1])/self.Delta[1],
        (Vbig[1:-1,1:-1,1:-1]-Vbig[1:-1,1:-1,:-2])/self.Delta[2]]
        u_int = self.B[0]*VB[0]+self.B[1]*VB[1]+self.B[2]*VB[2]
        m = np.minimum(-Ax[0]/self.B[0],-Ax[1]/self.B[1],-Ax[2]/self.B[2])
        return np.minimum(u_int,m)

    def solve_PFI(self):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.CF)
        while i < 10 and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs((V1 - V)))
            V, i = V1, i+1
            if np.min(V1-V) < 0:
                print("Failure of monotonicity at:", len(V[V1-V<0]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<0]))
            print("Difference between iterations:", eps, "Iterations:", i)
        return V

    def MPFI(self,u,V,M):
        norm = (self.rho + sum(self.tran_func(self.mesh([0,0,0]),u).values())).reshape((self.M,))
        b = (self.obj(self.xx,u) + self.BND(u)).reshape((self.M,))
        V = V.reshape((self.M,))
        H = self.H(u)
        for i in range(M+1):
            V = (b/norm) + H*V
        return V.reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def Update_MPFI(self,V,M):
        return self.MPFI(self.polupdate(V),V,M)

    def solve_MPFI(self,M):
        V, i, eps = -self.mat_quad(self.Q/self.rho,self.xx)/2, 1, 1
        V = self.V(0*self.CF)
        while i < self.maxiter and eps > self.tol:
            V1 = self.Update_MPFI(V,M)
            eps = np.amax(np.abs(V1 - V))
            print("Difference between iterations:", eps, "Iterations:", i)
            if np.min(V1-V) < -10**(-4):
                print("Failure of monotonicity at:", len(V[V1-V<-10**(-4)]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-10**(-4)]))
            V, i = V1, i+1
        return V

    def BND(self,u):
        CF = self.CF_fun(self.big_xx)
        BND = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        mesh12 = np.meshgrid(range(0,self.N[1]-1), range(0,self.N[2]-1), indexing='ij')
        mesh02 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[2]-1), indexing='ij')
        mesh01 = np.meshgrid(range(0,self.N[0]-1), range(0,self.N[1]-1), indexing='ij')
        p = self.tran_func((0,mesh12[0],mesh12[1]),u)
        BND[0,:,:] = p[(-1,0,0)]*CF[0,1:-1,1:-1]
        p = self.tran_func((self.N[0]-2,mesh12[0],mesh12[1]),u)
        BND[-1,:,:] = p[(1,0,0)]*CF[-1,1:-1,1:-1]
        p = self.tran_func((mesh02[0],0,mesh02[1]),u)
        BND[:,0,:] = BND[:,0,:] + p[(0,-1,0)]*CF[1:-1,0,1:-1]
        p = self.tran_func((mesh02[0],self.N[1]-2,mesh02[1]),u)
        BND[:,-1,:] = BND[:,-1,:] + p[(0,1,0)]*CF[1:-1,-1,1:-1]
        p = self.tran_func((mesh01[0],mesh01[1],0),u)
        BND[:,:,0] = BND[:,:,0] + p[(0,0,-1)]*CF[1:-1,1:-1,0]
        p = self.tran_func((mesh01[0],mesh01[1],self.N[2]-2),u)
        BND[:,:,-1] = BND[:,:,-1] + p[(0,0,1)]*CF[1:-1,1:-1,-1]
        return BND

    def mat_quad(self,M,z):
        return M[0,0]*z[0]*z[0] + M[0,1]*z[1]*z[0] + M[0,2]*z[2]*z[0] \
        + M[1,0]*z[0]*z[1] + M[1,1]*z[1]*z[1] + M[1,2]*z[2]*z[1] \
        + M[2,0]*z[0]*z[2] + M[2,1]*z[1]*z[2] + M[2,2]*z[2]*z[2]

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1] + M[0,2]*z[2], \
        M[1,0]*z[0] + M[1,1]*z[1] + M[1,2]*z[2], \
        M[2,0]*z[0] + M[2,1]*z[1] + M[2,2]*z[2]

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def mesh(self,m):
        return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
        range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), \
        range(max(-m[2],0), self.N[2] - 1 - max(m[2],0)), indexing='ij')
