"""
Classes for uncontrolled linear-quadratic problems. Correlated noise.

Class constructors here: LQ_degen3, LQ_degen5 and BOZ.
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

"""
3-point approximation
"""

class LQ_degen3(object):
    def __init__(self,Q=np.eye(2),sigma=0.25,rho=1,N=(100,100),pbar=0.001,mbar=4,x_bnd=[[0,1],[0,1]]):
        self.Q, self.sigma, self.rho, self.pbar = Q, sigma, rho, pbar
        self.N, self.M, self.mbar, self.x_bnd = N, (N[0]-1)*(N[1]-1), mbar, x_bnd
        self.Delta = (x_bnd[0][1] - x_bnd[0][0])/N[0], (x_bnd[1][1] - x_bnd[1][0])/N[1]

    def CF_fun(self,x):
        return -self.mat_quad(self.Q/(self.rho - self.sigma**2),x)/2

    def P_tran(self):
        opt_m = self.opt_m()[0]
        int_m = opt_m[0][1:-1,1:-1], opt_m[1][1:-1,1:-1]
        ii, jj = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        ii_big, jj_big = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        row, row_big = ii*(self.N[1]-1)+jj, ii_big*(self.N[1]-1)+jj_big
        #diagonal:
        P = self.P_func(row_big,row_big,1-2*self.pbar + 0*row_big)
        #m transitions at interior points: (only part that differs from 5-pt approx)
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii+int_m[0])*(self.N[1]-1)+jj+int_m[1],self.pbar+0*jj)
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii-int_m[0])*(self.N[1]-1)+jj-int_m[1],self.pbar+0*jj)
        #x[0] = low/high edges. lower left block, and upper right block.
        row = np.arange(1,self.N[1]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+(self.N[1]-1))),shape=(self.M,self.M))
        row = np.arange(1,self.N[1]-2) + (self.N[1]-1)*(self.N[0]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-(self.N[1]-1))),shape=(self.M,self.M))
        #x[1] = low/high, shifting up/down in x[1] direction.
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+1)),shape=(self.M,self.M))
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1) + self.N[1]-2
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-1)),shape=(self.M,self.M))
        #corners: (x[0],x[1]) = (low, low), (high, low), (low, high), (high, high)
        #contributions from heading IN from boundary. keep row in mind when writing column.
        row_bnd = np.array([0,self.N[1]-2,self.M-1-(self.N[1]-2),self.M-1])
        column_bnd = np.array([(self.N[1]-1)+1, (self.N[1]-2)+self.N[1]-1-1, \
        self.M-1-(self.N[1]-2)-(self.N[1]-1)+1, self.M-1-(self.N[1]-1)-1])
        return P + sp.coo_matrix((self.pbar+0*row_bnd,(row_bnd,column_bnd)),shape=(self.M,self.M))

    def opt_m(self):
        opt_m = (np.zeros((self.N[0]-1,self.N[1]-1)),np.zeros((self.N[0]-1,self.N[1]-1)))
        int_m = (np.zeros((self.N[0]-3,self.N[1]-3)),np.zeros((self.N[0]-3,self.N[1]-3)))
        ind = [(i,j) for i,j in itertools.product(range(self.mbar),range(self.mbar)) if (i,j) != (0,0)]
        w = self.sig_grid(2)[0]/self.sig_grid(2)[1]
        val1 = [np.abs(j-i/w).reshape((self.N[0]-3)*(self.N[1]-3),) for i,j in ind]
        I = np.argmin(np.array(val1), axis=0).reshape((self.N[0]-3,self.N[1]-3))
        for n in range(len(ind)):
            int_m[0][I==n],int_m[1][I==n] = ind[n][0], ind[n][1]
        opt_m[0][1:-1,1:-1] = self.bound_adj(int_m)[0]
        opt_m[1][1:-1,1:-1] = self.bound_adj(int_m)[1]
        bnd_m = self.bnd_m()
        m = opt_m[0] + bnd_m[0], opt_m[1] + bnd_m[1]
        return m, m[0] > m[1]

    def bnd_m(self):
        bnd = (np.zeros((self.N[0]-1,self.N[1]-1)), np.zeros((self.N[0]-1,self.N[1]-1)))
        #edges:
        bnd[0][1:-1,0], bnd[1][1:-1,0] = 0,1
        bnd[0][1:-1,-1], bnd[1][1:-1,-1] = 0,1
        bnd[0][0,1:-1], bnd[1][0,1:-1] = 1,0
        bnd[0][-1,1:-1], bnd[1][-1,1:-1] = 1,0
        #corners: no change with conventions
        bnd[0][0,0], bnd[1][0,0] = 1,1
        bnd[0][-1,-1], bnd[1][-1,-1] = 1,1
        bnd[0][-1,0], bnd[1][-1,0] = 1,1
        bnd[0][0,-1], bnd[1][0,-1] = 1,1
        return bnd

    def bnd(self):
        CF, BND = self.CF_fun(self.grid(0)), np.zeros((self.N[0]-1,self.N[1]-1))
        BND[1:-1,0], BND[1:-1,-1] = self.pbar*CF[2:-2,0], self.pbar*CF[2:-2,-1]
        BND[0,1:-1], BND[-1,1:-1] = self.pbar*CF[0,2:-2], self.pbar*CF[-1,2:-2]
        BND[0, 0], BND[0,-1] = self.pbar*CF[0,0], self.pbar*CF[0,-1]
        BND[-1,0], BND[-1,-1] = self.pbar*CF[-1,0], self.pbar*CF[-1,-1]
        return BND

    #notice that the Dt is using the CORRECT Dt terms.
    def Vsolve(self):
        m,ind = self.opt_m()
        Dt = ind*(2*self.pbar*m[0]**2/self.sig_grid(1)[0]**2) + (1-ind)*(2*self.pbar*m[1]**2/self.sig_grid(1)[1]**2)
        D = sp.diags(np.exp(-self.rho*Dt).reshape((self.M,)),0)
        b = -Dt*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*Dt)*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        tic = time.time()
        z = sp.linalg.spsolve(B, b.reshape((self.M,1))).reshape((self.N[0]-1,self.N[1]-1))
        toc = time.time()
        #print("Time to solve linear system for 3-pt, with grid size", self.N, "and steps", self.mbar, ":", toc-tic)
        return z

    def Bb(self):
        m,ind = self.opt_m()
        Dt = ind*(2*self.pbar*m[0]**2/self.sig_grid(1)[0]**2) + (1-ind)*(2*self.pbar*m[1]**2/self.sig_grid(1)[1]**2)
        D = sp.diags(np.exp(-self.rho*Dt).reshape((self.M,)),0)
        b = -Dt*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*Dt)*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        return B,b

    def bound_adj(self,m):
        i,j = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        return np.minimum(m[0],np.minimum(self.mbar,np.minimum(i,self.N[0]-2-i))), \
        np.minimum(m[1],np.minimum(self.mbar,np.minimum(j,self.N[1]-2-j)))

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def grid(self,m):
        x1 = np.linspace(self.x_bnd[0][0]+m*self.Delta[0],self.x_bnd[0][0]+(self.N[0]-m)*self.Delta[0],self.N[0]+1-2*m)
        x2 = np.linspace(self.x_bnd[1][0]+m*self.Delta[1],self.x_bnd[1][0]+(self.N[1]-m)*self.Delta[1],self.N[1]+1-2*m)
        return np.meshgrid(x1,x2,indexing='ij')

    def sig_grid(self,m):
        return self.sigma*self.grid(m)[0]/self.Delta[0],self.sigma*self.grid(m)[1]/self.Delta[1]

    def mat_quad(self,M,z):
        return M[0,0]*z[0]**2 + M[0,1]*z[1]*z[0] + M[1,0]*z[1]*z[0] + M[1,1]*z[1]**2

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1], M[1,0]*z[0] + M[1,1]*z[1]

"""
5-point approximation no drift. This sends the Dt to zero.
"""

class LQ_degen5(object):
    def __init__(self,Q=np.eye(2),sigma=0.25,rho=.1,N=(100,100),pbar=0.001,mbar=4,x_bnd=[[0.,1],[0.,1]]):
        self.Q, self.sigma, self.rho, self.pbar = Q, sigma, rho, pbar
        self.N, self.M, self.mbar, self.x_bnd = N, (N[0]-1)*(N[1]-1), mbar, x_bnd
        self.Delta = (x_bnd[0][1] - x_bnd[0][0])/N[0], (x_bnd[1][1] - x_bnd[1][0])/N[1]
        self.index = [(i,j) for i,j in itertools.product(range(self.mbar),range(self.mbar)) if (i,j)!=(0,0)]

    def CF_fun(self,x):
        return -self.mat_quad(self.Q/(self.rho - self.sigma**2),x)/2

    #(ii_big, jj_big) are grids that include points adjacent to boundary
    def P_tran(self):
        m, prob = self.opt_m()
        ii, jj = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        (m1, m2), prob = m, (prob[0][ii-1,jj-1],prob[1][ii-1,jj-1])
        ii_big, jj_big = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        row, row_big = ii*(self.N[1]-1)+jj, ii_big*(self.N[1]-1)+jj_big
        P = self.P_func(row_big,row_big,1-2*self.pbar + 0*row_big)
        #(N[0]-1)**2 blocks of size (N[1]-1)x(N[1]-1)
        #Now the transitions on the interior (smallest) grid.
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii+m1[0])*(self.N[1]-1)+jj+m1[1],prob[0])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii+m2[0])*(self.N[1]-1)+jj+m2[1],prob[1])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii-m1[0])*(self.N[1]-1)+jj-m1[1],prob[0])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii-m2[0])*(self.N[1]-1)+jj-m2[1],prob[1])
        #x[0] = low/high edges. lower left block, and upper right block.
        row = np.arange(1,self.N[1]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+(self.N[1]-1))),shape=(self.M,self.M))
        row = np.arange(1,self.N[1]-2) + (self.N[1]-1)*(self.N[0]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-(self.N[1]-1))),shape=(self.M,self.M))
        #x[1] = low/high, shifting up/down in x[1] direction.
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+1)),shape=(self.M,self.M))
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1) + self.N[1]-2
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-1)),shape=(self.M,self.M))
        #corners: (x[0],x[1]) = (low, low), (high, low), (low, high), (high, high)
        #contributions from heading IN from boundary. keep row in mind when writing column.
        row_bnd = np.array([0,self.N[1]-2,self.M-1-(self.N[1]-2),self.M-1])
        column_bnd = np.array([(self.N[1]-1)+1, (self.N[1]-2)+self.N[1]-1-1, \
        self.M-1-(self.N[1]-2)-(self.N[1]-1)+1, self.M-1-(self.N[1]-1)-1])
        return P + sp.coo_matrix((self.pbar+0*row_bnd,(row_bnd,column_bnd)),shape=(self.M,self.M))

    #remember, only selected on the smallest grid self.grid(2).
    def opt_m(self):
        #preallocation candidate
        m1 = (np.zeros((self.N[0]-3,self.N[1]-3)),np.zeros((self.N[0]-3,self.N[1]-3)))
        m2 = (np.zeros((self.N[0]-3,self.N[1]-3)),np.zeros((self.N[0]-3,self.N[1]-3)))
        M_int, w = (self.N[1]-3)*(self.N[0]-3), self.sig_grid(2)[0]/self.sig_grid(2)[1]
        val = [(np.minimum(1,w)*np.abs(j-i/w)).reshape(M_int,) for i,j in self.index]
        I = np.argmin(np.array(val),axis=0).reshape((self.N[0]-3,self.N[1]-3))
        ind = (w>1).reshape((self.N[0]-3,self.N[1]-3)) #not needed
        #get first pair:
        for n in range(len(self.index)):
            m1[0][I==n],m1[1][I==n] = self.index[n]
        z = np.minimum(1,w)*(m1[1]-m1[0]/w)
        #z = 0*z #testing that this should reduce to 3-point
        #get second pair, then ensure it remains on grid:
        m2 = m1[0] + (w<=1)*(2*(z>0) - 1), m1[1] + (w>1)*(2*(z<=0) - 1)
        #m2 = m1[0], m1[1] #testing that this should reduce to 3-point
        return (self.bound_adj(m1),self.bound_adj(m2)), (self.pbar*(1-np.abs(z)),self.pbar*np.abs(z))

    #following describes the terms that must be added to flow
    def bnd_m(self):
        bnd = (np.zeros((self.N[0]-1,self.N[1]-1)), np.zeros((self.N[0]-1,self.N[1]-1)))
        #edges:
        bnd[0][1:-1,0], bnd[1][1:-1,0] = 0,1
        bnd[0][1:-1,-1], bnd[1][1:-1,-1] = 0,1
        bnd[0][0,1:-1], bnd[1][0,1:-1] = 1,0
        bnd[0][-1,1:-1], bnd[1][-1,1:-1] = 1,0
        #corners: no change with conventions
        bnd[0][0,0], bnd[1][0,0] = 1,1
        bnd[0][-1,-1], bnd[1][-1,-1] = 1,1
        bnd[0][-1,0], bnd[1][-1,0] = 1,1
        bnd[0][0,-1], bnd[1][0,-1] = 1,1
        return bnd

    def bnd(self):
        CF, BND = self.CF_fun(self.grid(0)), np.zeros((self.N[0]-1,self.N[1]-1))
        BND[1:-1,0], BND[1:-1,-1] = self.pbar*CF[2:-2,0], self.pbar*CF[2:-2,-1]
        BND[0,1:-1], BND[-1,1:-1] = self.pbar*CF[0,2:-2], self.pbar*CF[-1,2:-2]
        BND[0, 0], BND[0,-1] = self.pbar*CF[0,0], self.pbar*CF[0,-1]
        BND[-1,0], BND[-1,-1] = self.pbar*CF[-1,0], self.pbar*CF[-1,-1]
        return BND

    def Dt(self):
        (m1,m2), p = self.opt_m()
        big_m = self.bnd_m()
        big_m[0][1:-1,1:-1],big_m[1][1:-1,1:-1] = m1[0],m1[1]
        ind = self.sig_grid(1)[0] > self.sig_grid(1)[1]
        Dt = 2*self.pbar*(ind*big_m[0]**2/self.sig_grid(1)[0]**2 \
        + (1-ind)*big_m[1]**2/self.sig_grid(1)[1]**2)
        return Dt

    def Vsolve(self):
        D = sp.diags(np.exp(-self.rho*self.Dt()).reshape((self.M,)),0)
        b = -self.Dt()*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*self.Dt())*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        #print("Type of matrices:", type(B), type(b))
        tic = time.time()
        z = sp.linalg.spsolve(B, b.reshape((self.M,1))).reshape((self.N[0]-1,self.N[1]-1))
        toc = time.time()
        print("Time to solve linear system for 5-pt, with grid size", self.N, "and steps", self.mbar, ":", toc-tic)
        return z

    def Bb(self):
        D = sp.diags(np.exp(-self.rho*self.Dt()).reshape((self.M,)),0)
        b = -self.Dt()*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*self.Dt())*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        return B,b

    def bound_adj(self,m):
        i,j = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        return np.minimum(m[0],np.minimum(self.mbar,np.minimum(i,self.N[0]-2-i))), \
        np.minimum(m[1],np.minimum(self.mbar,np.minimum(j,self.N[1]-2-j)))

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def grid(self,k):
        x1 = np.linspace(self.x_bnd[0][0]+k*self.Delta[0],self.x_bnd[0][0]+(self.N[0]-k)*self.Delta[0],self.N[0]+1-2*k)
        x2 = np.linspace(self.x_bnd[1][0]+k*self.Delta[1],self.x_bnd[1][0]+(self.N[1]-k)*self.Delta[1],self.N[1]+1-2*k)
        return np.meshgrid(x1,x2,indexing='ij')

    def sig_grid(self,m):
        return self.sigma*self.grid(m)[0]/self.Delta[0],self.sigma*self.grid(m)[1]/self.Delta[1]

    def mat_quad(self,M,z):
        return M[0,0]*z[0]**2 + M[0,1]*z[1]*z[0] + M[1,0]*z[1]*z[0] + M[1,1]*z[1]**2

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1], M[1,0]*z[0] + M[1,1]*z[1]

"""
Bonnans, Ottonwaelter and Zidani.
"""

class BOZ(object):
    def __init__(self,Q=np.eye(2),sigma=0.25,rho=.1,N=(100,100),pbar=0.001,mbar=4,x_bnd=[[0.,1],[0.,1]]):
        self.Q, self.sigma, self.rho, self.pbar = Q, sigma, rho, pbar
        self.N, self.M, self.mbar, self.x_bnd = N, (N[0]-1)*(N[1]-1), mbar, x_bnd
        self.Delta = (x_bnd[0][1] - x_bnd[0][0])/N[0], (x_bnd[1][1] - x_bnd[1][0])/N[1]
        self.index = [(i,j) for i,j in itertools.product(range(self.mbar),range(self.mbar)) if (i,j)!=(0,0)]
        self.chi = self.sig_grid(2)[0] >= self.sig_grid(2)[1]
        self.Fah = np.empty((3,1),dtype=object)
        #following is F(ah) (image of ah under F)
        self.Fah[0,0], self.Fah[1,0], self.Fah[2,0] = self.sig_grid(2)[0]**2/2, \
        np.sqrt(2)*self.sig_grid(2)[0]*self.sig_grid(2)[1]/2, self.sig_grid(2)[1]**2/2

    def CF_fun(self,x):
        return -self.mat_quad(self.Q/(self.rho - self.sigma**2),x)/2

    def P_tran(self):
        m, eta = self.opt_m()
        ii, jj = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        (m1, m2), eta = m, (eta[0,0][ii-1,jj-1],eta[1,0][ii-1,jj-1])
        prob = (self.pbar*eta[0]/(eta[1] + eta[0]), self.pbar*eta[1]/(eta[1] + eta[0]))
        ii_big, jj_big = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        row, row_big = ii*(self.N[1]-1)+jj, ii_big*(self.N[1]-1)+jj_big
        P = self.P_func(row_big,row_big,1-2*self.pbar + 0*row_big)
        #(N[0]-1)**2 blocks of size (N[1]-1)x(N[1]-1)
        #Now the transitions on the interior (smallest) grid.
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii+m1[0])*(self.N[1]-1)+jj+m1[1],prob[0])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii+m2[0])*(self.N[1]-1)+jj+m2[1],prob[1])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii-m1[0])*(self.N[1]-1)+jj-m1[1],prob[0])
        P = P+self.P_func(ii*(self.N[1]-1)+jj,(ii-m2[0])*(self.N[1]-1)+jj-m2[1],prob[1])
        #x[0] = low/high edges. lower left block, and upper right block.
        row = np.arange(1,self.N[1]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+(self.N[1]-1))),shape=(self.M,self.M))
        row = np.arange(1,self.N[1]-2) + (self.N[1]-1)*(self.N[0]-2)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-(self.N[1]-1))),shape=(self.M,self.M))
        #x[1] = low/high, shifting up/down in x[1] direction.
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1)
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row+1)),shape=(self.M,self.M))
        row = np.arange(1,self.N[0]-2)*(self.N[1]-1) + self.N[1]-2
        P = P + sp.coo_matrix((self.pbar+0*row,(row,row-1)),shape=(self.M,self.M))
        #corners: (x[0],x[1]) = (low, low), (high, low), (low, high), (high, high)
        #contributions from heading IN from boundary. keep row in mind when writing column.
        row_bnd = np.array([0,self.N[1]-2,self.M-1-(self.N[1]-2),self.M-1])
        column_bnd = np.array([(self.N[1]-1)+1, (self.N[1]-2)+self.N[1]-1-1, \
        self.M-1-(self.N[1]-2)-(self.N[1]-1)+1, self.M-1-(self.N[1]-1)-1])
        return P + sp.coo_matrix((self.pbar+0*row_bnd,(row_bnd,column_bnd)),shape=(self.M,self.M))

    def opt_m(self):
        m = [(np.zeros((self.N[0]-3,self.N[1]-3)),np.zeros((self.N[0]-3,self.N[1]-3))) for _ in range(2)]
        eta = [np.zeros((self.N[0]-3,self.N[1]-3)) for _ in range(2)]
        ii, jj = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')  #interior grid
        (p, q), (p1, q1) = (self.chi, 1 + 0*self.chi), (1 + 0*self.chi, 1-self.chi)
        rat = (p, q), (p1, q1)
        for i in range(self.mbar):
            stop = self.stop(rat)
            #should the following be squared? NO. Checked and correct.
            #p_true, q_true = (self.sigma*self.grid(2)[0]/self.Delta[0])**2, (self.sigma*self.grid(2)[1]/self.Delta[1])**2
            p_true, q_true = self.sigma*self.grid(2)[0]/self.Delta[0], self.sigma*self.grid(2)[1]/self.Delta[1]
            (p, q), (p1, q1) = rat
            (p11, q11) = (p + p1, q + q1)
            #indicator for if child is steeper than true:
            ind0 = self.spec_divide(q11,p11) >= self.spec_divide(q_true,p_true)
            #always keep true between p,q and p1,q1:
            p = stop*p + (1-stop)*(ind0*p11 + (1-ind0)*p)
            q = stop*q + (1-stop)*(ind0*q11 + (1-ind0)*q)
            p1 = stop*p1 + (1-stop)*(ind0*p1 + (1-ind0)*p11)
            q1 = stop*q1 + (1-stop)*(ind0*q1 + (1-ind0)*q11)
            rat = (p, q), (p1, q1)
        aprime = self.a_mult(self.proj(((p, q), (p1, q1))),self.Fah)
        A, b = np.empty((2,2),dtype=object), np.empty((2,1),dtype=object)
        b[0,0], b[1,0] = aprime[0,0], aprime[2,0]
        A[0,0], A[0,1], A[1,0], A[1,1] = p**2, p1**2, q**2, q1**2
        eta = self.a_mult(self.a_inv2D(A),b)
        eta[0,0], eta[1,0] = np.maximum(eta[0,0],0), np.maximum(eta[1,0],0)
        return (self.bound_adj((p,q)),self.bound_adj((p1,q1))), eta

    def proj(self,rat):
        (p, q), (p1, q1) = rat
        A = np.empty((3,2),dtype=object)
        A[0,0], A[0,1] = p**2, p1**2
        A[1,0], A[1,1] = np.sqrt(2)*p*q, np.sqrt(2)*p1*q1
        A[2,0], A[2,1] = q**2, q1**2
        M = self.a_mult(A,self.a_inv2D(self.a_mult(self.a_tran(A),A)))
        return self.a_mult(M,self.a_tran(A))

    def stop(self,rat):
        i,j = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        (p, q), (p1, q1) = rat
        aprime = self.a_mult(self.proj(rat),self.Fah)
        diffnorm = np.sqrt(sum([(self.Fah[i,0] - aprime[i,0])**2 for i in range(3)]))
        norm = np.sqrt(sum([self.Fah[i,0]**2 for i in range(3)]))
        big = p + p1 > np.minimum(self.mbar,np.minimum(i,self.N[0]-2-i)), \
        q + q1 > np.minimum(self.mbar,np.minimum(j,self.N[1]-2-j))
        return np.minimum(np.minimum( np.maximum(big[0],big[1]), 1), 1)

    def bnd_m(self):
        bnd = (np.zeros((self.N[0]-1,self.N[1]-1)), np.zeros((self.N[0]-1,self.N[1]-1)))
        #edges:
        bnd[0][1:-1,0], bnd[1][1:-1,0] = 0,1
        bnd[0][1:-1,-1], bnd[1][1:-1,-1] = 0,1
        bnd[0][0,1:-1], bnd[1][0,1:-1] = 1,0
        bnd[0][-1,1:-1], bnd[1][-1,1:-1] = 1,0
        #corners: no change with conventions
        bnd[0][0,0], bnd[1][0,0] = 1,1
        bnd[0][-1,-1], bnd[1][-1,-1] = 1,1
        bnd[0][-1,0], bnd[1][-1,0] = 1,1
        bnd[0][0,-1], bnd[1][0,-1] = 1,1
        return bnd

    def bnd(self):
        CF, BND = self.CF_fun(self.grid(0)), np.zeros((self.N[0]-1,self.N[1]-1))
        BND[1:-1,0], BND[1:-1,-1] = self.pbar*CF[2:-2,0], self.pbar*CF[2:-2,-1]
        BND[0,1:-1], BND[-1,1:-1] = self.pbar*CF[0,2:-2], self.pbar*CF[-1,2:-2]
        BND[0, 0], BND[0,-1] = self.pbar*CF[0,0], self.pbar*CF[0,-1]
        BND[-1,0], BND[-1,-1] = self.pbar*CF[-1,0], self.pbar*CF[-1,-1]
        return BND

    def Dt(self):
        (m1,m2), eta = self.opt_m()
        big_m = self.bnd_m()
        big_m[0][1:-1,1:-1],big_m[1][1:-1,1:-1] = m1[0],m1[1]
        ind = self.sig_grid(1)[0] > self.sig_grid(1)[1]
        Dt = 2*self.pbar*(ind*big_m[0]**2/self.sig_grid(1)[0]**2 \
        + (1-ind)*big_m[1]**2/self.sig_grid(1)[1]**2)
        Dt[1:-1,1:-1] = self.pbar/(eta[0,0] + eta[1,0])
        return Dt

    def Vsolve(self):
        D = sp.diags(np.exp(-self.rho*self.Dt()).reshape((self.M,)),0)
        b = -self.Dt()*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*self.Dt())*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        tic = time.time()
        z = sp.linalg.spsolve(B, b.reshape((self.M,1))).reshape((self.N[0]-1,self.N[1]-1))
        toc = time.time()
        #print("Time to solve linear system for BOZ, with grid size", self.N, "and steps", self.mbar, ":", toc-tic)
        return z

    def Bb(self):
        D = sp.diags(np.exp(-self.rho*self.Dt()).reshape((self.M,)),0)
        b = -self.Dt()*self.mat_quad(self.Q,self.grid(1))/2 + np.exp(-self.rho*self.Dt())*self.bnd()
        B = sp.eye(self.M) - D*self.P_tran()
        B,b = 10**2*B,10**2*b
        return B,b

    def bound_adj(self,m):
        i,j = np.meshgrid(range(1,self.N[0]-2),range(1,self.N[1]-2),indexing='ij')
        return np.minimum(m[0],np.minimum(self.mbar,np.minimum(i,self.N[0]-2-i))), \
        np.minimum(m[1],np.minimum(self.mbar,np.minimum(j,self.N[1]-2-j)))

    def P_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def grid(self,k):
        x1 = np.linspace(self.x_bnd[0][0]+k*self.Delta[0],self.x_bnd[0][0]+(self.N[0]-k)*self.Delta[0],self.N[0]+1-2*k)
        x2 = np.linspace(self.x_bnd[1][0]+k*self.Delta[1],self.x_bnd[1][0]+(self.N[1]-k)*self.Delta[1],self.N[1]+1-2*k)
        return np.meshgrid(x1,x2,indexing='ij')

    def sig_grid(self,m):
        return self.sigma*self.grid(m)[0]/self.Delta[0],self.sigma*self.grid(m)[1]/self.Delta[1]

    def a_diff(self,A,B):
        (m,n) = np.shape(A)
        diff = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                diff[i,j] = A[i,j] - B[i,j]
        return diff

    def a_mult(self,A,B):
        (m1, n1), (m2, n2) = A.shape, B.shape
        AB = np.empty((m1, n2), dtype=object)
        for i in range(m1):
            for j in range(n2):
                AB[i,j] = sum([A[i,k]*B[k,j] for k in range(n1)])
        return AB

    def a_tran(self,A):
        (m, n) = A.shape
        AT = np.empty((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                AT[i,j] = A[j,i]
        return AT

    def a_inv2D(self,A):
        (m, n) = A.shape
        Ainv = np.empty((2, 2), dtype=object)
        det = A[1,1]*A[0,0] - A[0,1]*A[1,0]
        Ainv[0,0], Ainv[1,0] = A[1,1]/det, -A[1,0]/det
        Ainv[0,1], Ainv[1,1] = -A[0,1]/det, A[0,0]/det
        return Ainv

    def spec_divide(self,a,b):
        with np.errstate(divide='ignore',invalid='ignore'):
            x = a/b
        x[b==0] = 100000
        return x

    def mat_quad(self,M,z):
        return M[0,0]*z[0]**2 + M[0,1]*z[1]*z[0] + M[1,0]*z[1]*z[0] + M[1,1]*z[1]**2

    def mat_array(self,M,z):
        return M[0,0]*z[0] + M[0,1]*z[1], M[1,0]*z[0] + M[1,1]*z[1]
