"""
Macrofinance class constructors: MF_ind, MF_corr and MF_corr_var_dt (for a check).
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as scopt
from scipy.sparse import linalg
from scipy.sparse import diags
import scipy.sparse.linalg as splinalg
import itertools, time

"""
Independent noise
"""

class MF_ind(object):
    def __init__(self,rho=[.15, .1],gamma=0.75,Pi=0.05,rlow=0.0, sigsigbar=0.2,
    theta=1, N=(200,50), X_bnd=[[0,1],[0.1,0.4]],tol=10**-6,dt=10**(-8),
    Delta_y=0.1, max_iter_eq=500, pol_maxiter = 20, relax = 0.0):
        self.rho, self.gamma, self.Pi, self.rlow = rho, gamma, Pi, rlow
        self.theta, self.sigsigbar, self.X_bnd = theta, sigsigbar, X_bnd
        self.sigbar = (self.X_bnd[1][1] + self.X_bnd[1][0])/2
        self.max_iter_eq, self.N, self.M = max_iter_eq, N, (N[0]+1)*(N[1]+1)
        self.tol, self.dt, self.Delta_y = tol, dt, Delta_y
        self.Delta = (self.X_bnd[0][1]-self.X_bnd[0][0])/self.N[0], (self.X_bnd[1][1]-self.X_bnd[1][0])/self.N[1]
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.XX, self.SIGSIG = self.grid(0)
        self.sigsig = self.sigsigbar*(self.SIGSIG<self.X_bnd[1][1] - 2*self.Delta[1])*(self.SIGSIG>self.X_bnd[1][0] + 2*self.Delta[1])
        self.relax, self.pol_maxiter = relax, pol_maxiter

    def polupdate(self,type,V,agg):
        E = self.con_E()
        VFx, VBx = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VFx[:-1,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        VBx[1:,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        Vpx = (self.gamma<1)*VFx + (self.gamma>1)*VBx
        k1 = (self.Pi - agg[0])/(self.gamma*self.SIGSIG**2) + agg[2]*self.XX*Vpx/(self.gamma*self.SIGSIG*V*E[0])
        k = (E[0]/E[1])*np.maximum(k1,0)
        c = (self.rho[type]/V)**(1/self.gamma)*np.exp(self.rho[type]*self.dt/self.gamma)*E[2]
        return c, (type==0)*k + (type==1)*0

    def p_func(self,ind,type,pol,agg):
        (c,k), (r, mux, sigx) = pol, agg
        (ii,jj), p_func = ind, {}
        pup_y = (self.dt/self.Delta_y**2)*(self.SIGSIG**2*k**2/2 + self.Delta_y*(r + (self.Pi-r)*k))
        pdown_y = (self.dt/self.Delta_y**2)*(self.SIGSIG**2*k**2/2 + self.Delta_y*(c + self.SIGSIG**2*k**2/2))
        d = [self.dt/self.Delta[i]**2 for i in range(2)]
        muK = self.XX*sigx*(1-self.gamma)*self.SIGSIG*k
        pup_x = d[0]*((sigx*self.XX)**2/2 + self.Delta[0]*(np.maximum(mux*self.XX,0)+np.maximum(muK,0)))
        pdown_x = d[0]*((sigx*self.XX)**2/2 + self.Delta[0]*(np.maximum(-mux*self.XX,0)+np.maximum(-muK,0)))
        pup_sig = d[1]*(self.sigsig**2/2 + self.Delta[1]*np.maximum(self.theta*(self.sigbar-self.SIGSIG),0))
        pdown_sig = d[1]*(self.sigsig**2/2 + self.Delta[1]*np.maximum(-self.theta*(self.sigbar-self.SIGSIG),0))
        p_func['up'], p_func['down'] = pup_y[ii,jj], pdown_y[ii,jj]
        p_func[(1,0)], p_func[(-1,0)] = pup_x[ii,jj], pdown_x[ii,jj]
        p_func[(0,1)], p_func[(0,-1)] = pup_sig[ii,jj], pdown_sig[ii,jj]
        return p_func

    def H(self,type,pol,agg):
        (c, k), rho = pol, self.rho[type]
        ii, jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),indexing = 'ij')
        p, e = self.p_func((ii,jj),type,pol,agg), np.exp((1-self.gamma)*self.Delta_y)
        diag = 1 + p['up']*(e-1) + p['down']*(1/e-1) - p[(1,0)] - p[(-1,0)] - p[(0,1)] - p[(0,-1)]
        H = self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj,diag)
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)),indexing='ij')
            H = H + self.T_func(ii*(self.N[1]+1)+jj,(ii+key[0])*(self.N[1]+1)+jj+key[1],self.p_func((ii,jj),type,pol,agg)[key])
        return H

    def T(self,type,pol,agg):
        return (1/self.dt)*(np.exp(-self.rho[type]*self.dt)*self.H(type,pol,agg) - sp.eye(self.M))/(1-self.gamma)

    def Vupdate_PFI(self,type,pol,agg):
        b = self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        return sp.linalg.spsolve(-self.T(type,pol,agg), b.reshape((self.M,1))).reshape((self.N[0]+1,self.N[1]+1))

    def updateV_PFI(self,type,V,agg):
        return self.Vupdate_PFI(type,self.polupdate(type,V,agg),agg)

    def solveV_PFI(self,type,agg):
        c,k = self.rho[type]/10 + 0*agg[0], 0*agg[0]
        V = self.Vupdate_PFI(type,(c,k),agg)
        (c,k) = self.polupdate(type,V,agg)
        check_NaN = np.isnan(agg[0]).any()+np.isnan(agg[1]).any()+np.isnan(agg[2]).any()
        eps, eps2, i = 1, 1, 1
        if check_NaN==1:
            print("ERROR: NaNs in equilibrium quantities")
        while i < 40 and eps > self.tol:
            V1 = self.Vupdate_PFI(type,(c,k),agg)
            (c1,k1) = self.polupdate(type,V1,agg)
            eps = np.mean(np.abs(V**(1/(1-self.gamma)) - V1**(1/(1-self.gamma))))
            eps2 = np.max(np.abs(c - c1)) + np.max(np.abs(k - k1))
            if np.min(V1/(1-self.gamma)-V/(1-self.gamma)) < -10**(-4):
                diff = V1/(1-self.gamma)-V/(1-self.gamma)
                print("Failure of monotonicity at:", len(V[diff<-10**(-4)]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean(diff[diff<-10**(-4)]))
            if np.min(V1) < 0:
                print("ERROR: value function becomes negative")
            V, (c,k), i = V1, (c1,k1), i+1
        if eps > self.tol or np.isnan(V).any() or np.min(V1)<0:
            print("Individual problem did not converge", "Difference:", eps)
        else:
            print("Convergence of individual problem in", i, "iterations.", "Difference:", eps)
        return V

    def Vupdate(self,type,pol,V,M,agg):
        b = self.dt*self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        V, T = V.reshape((self.M,1)), self.T(type,pol,agg)
        for i in range(M+1):
            V = V + (1-self.gamma)*(b.reshape((self.M,1)) + self.dt*T*V)
        return V.reshape((self.N[0]+1,self.N[1]+1))

    def updateV(self,type,V,agg,M):
        return self.Vupdate(type,self.polupdate(type,V,agg),V,M,agg)

    def agg_update(self,V_pair):
        E, VE = self.con_E(), V_pair[0]
        VFx, VBx = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VFx[:-1,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        VBx[1:,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        Vpx = (self.gamma<1)*VFx + (self.gamma>1)*VBx
        denom = self.gamma + (1-self.XX)*self.XX*(-Vpx)/(E[1]*VE)
        with np.errstate(divide='ignore',invalid='ignore'):
            sigx = self.SIGSIG*(1-self.XX)*np.maximum(np.minimum((E[0]/E[1])*(self.Pi -self.rlow)/(self.SIGSIG**2*denom), 1/self.XX), 0)
            r = np.maximum(self.rlow,self.Pi + self.SIGSIG*sigx*self.XX*Vpx/VE-self.gamma*self.SIGSIG**2/self.XX)
        r[self.XX==0], sigx[self.XX==0] = self.rlow, 0
        k1 = (self.Pi - r)/(self.gamma*self.SIGSIG**2) + sigx*self.XX*Vpx/(self.gamma*self.SIGSIG*VE*E[0])
        k = (E[0]/E[1])*np.maximum(k1,0)
        c = (self.rho[0]/V_pair[0])**(1/self.gamma)*np.exp(self.rho[0]*self.dt/self.gamma)*E[2], \
        (self.rho[1]/V_pair[1])**(1/self.gamma)*np.exp(self.rho[1]*self.dt/self.gamma)*E[2]
        mux = (c[1] - c[0] + (self.Pi-r)*k - self.SIGSIG**2*k**2*self.XX)*(1-self.XX)
        return r, mux, sigx

    def solve_FT(self,check):
        agg_PFI = self.solve_PFI()[0]
        if check==1:
            agg = agg_PFI
            V_pair = self.solveV_PFI(0,agg), self.solveV_PFI(1,agg)
        else:
            agg = self.log_quant()
            V_pair = 1 + 0*self.solveV_PFI(0,agg), 1+0*self.solveV_PFI(1,agg)
        eps, eps2, i = 1, 1, 1
        tic = time.time()
        while i < self.max_iter_eq and eps2 > 0*self.tol:
            r, mux, sigx = self.agg_update(V_pair)
            eps = np.amax(np.abs(r - agg[0])) + np.amax(np.abs((mux - agg[1])*self.XX)) + np.amax(np.abs((sigx - agg[2])*self.XX))
            eps2 = np.amax(np.abs(r - agg_PFI[0])) + np.amax(np.abs((mux - agg_PFI[1])*self.XX)) + np.amax(np.abs((sigx - agg_PFI[2])*self.XX))
            V_pair2 = (self.updateV(0,V_pair[0],(r, mux, sigx),0), self.updateV(1,V_pair[1],(r, mux, sigx),0))
            eps_V = np.amax(np.abs(V_pair2[0] - V_pair[0])) + np.amax(np.abs(V_pair2[1] - V_pair[1]))
            V_pair, agg, i = V_pair2, (r, mux, sigx), i+1
            print("Outer iteration:", i, "Differences:", eps,eps2)
        toc = time.time()
        print("Time taken for false transient:", toc-tic)
        return r, mux, sigx

    def solve_PFI(self):
        agg, eps, i = self.log_quant(), 1, 1
        V_pair = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
        tic = time.time()
        while i < self.pol_maxiter and eps > self.tol:
            r, mux, sigx = self.agg_update(V_pair)
            eps = np.amax(np.abs(r - agg[0])) + np.amax(np.abs(mux - agg[1])) + np.amax(np.abs(sigx - agg[2]))
            eps2 = np.mean(np.abs(r - agg[0])) + np.mean(np.abs(mux - agg[1])) + np.mean(np.abs(sigx - agg[2]))
            agg = r*(1-self.relax) + self.relax*agg[0], mux*(1-self.relax) \
            + self.relax*agg[1], sigx*(1-self.relax) + self.relax*agg[2]
            V_pair2 = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
            eps_V = np.amax(np.abs(V_pair2[0] - V_pair[0])) + np.amax(np.abs(V_pair2[1] - V_pair[1]))
            V_pair, i = V_pair2, i+1
            print("Outer iteration in PFI:", i, "Differences:", eps,eps2)
        if eps < self.tol:
            print("PFI converged in:", i, "iterations")
        else:
            print("PFI did not converge after:", i, "iterations")
            print("Difference:", eps)
        toc = time.time()
        return (r, mux, sigx), toc-tic

    def grid(self,m):
        x1 = np.linspace(self.X_bnd[0][0]+m*self.Delta[0],self.X_bnd[0][1]-m*self.Delta[0],self.N[0]+1-2*m)
        x2 = np.linspace(self.X_bnd[1][0]+m*self.Delta[1],self.X_bnd[1][1]-m*self.Delta[1],self.N[1]+1-2*m)
        return np.meshgrid(x1,x2,indexing='ij')

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def con_E(self):
        E1 = (np.exp((1-self.gamma)*self.Delta_y) - 1)/(self.Delta_y*(1-self.gamma))
        E2 = ((2 - np.exp(-(1-self.gamma)*self.Delta_y) - np.exp((1-self.gamma)*self.Delta_y))/self.Delta_y**2 \
        + (1 - np.exp(-(1-self.gamma)*self.Delta_y))/self.Delta_y)/(self.gamma*(1-self.gamma))
        Ec = (self.Delta_y*(1-self.gamma)/(1-np.exp(-(1-self.gamma)*self.Delta_y)))**(1/self.gamma)
        return E1, E2, Ec

    def log_quant(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            r_log = np.maximum(self.rlow, self.Pi - self.SIGSIG**2/self.XX)
            kbar = np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
            sigx_log = self.SIGSIG*(1-self.XX)*np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
        r_log[self.XX==0], sigx_log[self.XX==0], kbar[self.XX==0] = self.rlow, 0, 0
        mux_log = (self.rho[1] - self.rho[0] + (self.Pi-r_log)*kbar - self.SIGSIG**2*kbar**2*self.XX)*(1-self.XX)
        return r_log, mux_log, sigx_log

    def V_bnd(self,type,r):
        s = (type==0)*self.con_E()[0]*(self.Pi-r)**2/(2*self.con_E()[1]*self.SIGSIG**2*self.gamma)
        denom = (np.exp(self.rho[type]*self.dt) - 1)/self.dt - (1-self.gamma)*self.con_E()[0]*(r + s)
        if np.min(denom) < 0:
            print("Boundary values not well-defined")
        else:
            return self.gamma**self.gamma*self.rho[type]*np.exp(self.rho[type]*self.dt) \
            *self.con_E()[2]**(self.gamma*(1-self.gamma))*denom**(-self.gamma)

    def r_bnd(self,x):
        if x==0:
            return self.rlow
        else:
            return np.maximum(self.rlow,self.Pi - self.gamma*self.SIGSIG**2*(self.con_E()[0]/self.con_E()[1])**(-1))

    def V_bnd_eq(self,type,x):
        return self.V_bnd(type,self.r_bnd(x))

    def V_bnd_lin(self,type):
        return self.V_bnd_eq(type,0)*(1-self.XX) + self.V_bnd_eq(type,1)*self.XX

"""
Perfectly correlated noise (fixed timestep, nonlocal transition probabilities adjust)

Reminders:

    SIGSIG is array for sigma variable (volatility of depreciation shocks).
    sigbar is average sigma. sigsigbar is volatility of sigma.
    self.sigsig is array of sigsigbar (but set to zero on boundaries).
    mbar = size of largest non-local transition.
    mux and sigx are literally \mu_x and \sigma_x, not x\mu_x and x\sigma_x.

"""

class MF_corr(object):
    def __init__(self, rho=[.2, .1], gamma=0.9995, Pi=0.065, rlow=0.0, sigsigbar=0.2,
    theta=1, N=(200,50), X_bnd=[[0,1],[0.1,0.4]], mbar=4, tol=10**-6, dt=10**(-8),
    Delta_y=0.01, max_iter_eq=100, pol_maxiter = 20, relax = [0.0]):
        self.rho, self.gamma, self.Pi, self.rlow = rho, gamma, Pi, rlow
        self.theta, self.sigsigbar, self.X_bnd = theta, sigsigbar, X_bnd
        self.sigbar = (self.X_bnd[1][1] + self.X_bnd[1][0])/2
        self.N, self.M = N, (N[0]+1)*(N[1]+1)
        self.mbar, self.tol = mbar, tol
        self.tol, self.dt, self.Delta_y = tol, dt, Delta_y
        self.max_iter_eq = max_iter_eq
        self.Delta = (self.X_bnd[0][1]-self.X_bnd[0][0])/self.N[0], (self.X_bnd[1][1]-self.X_bnd[1][0])/self.N[1]
        self.index = [(i,j) for i,j in itertools.product(range(self.mbar),range(self.mbar)) if (i,j)!=(0,0)]
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.XX, self.SIGSIG = self.grid(0)
        self.sigsig = self.sigsigbar*(self.SIGSIG<self.X_bnd[1][1] - 0*self.Delta[1])*(self.SIGSIG>self.X_bnd[1][0] + 0*self.Delta[1])
        self.ii_, self.jj_ = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),indexing = 'ij')
        self.relax, self.pol_maxiter = relax, pol_maxiter

    #following produces dictionary of probabilities, given type, individual policy
    #functions, and aggregate quantities, at desired indices (points) ind.
    #slight difference from IFP and LQ due to homogeneity observation.
    def p_func(self,ind,type,pol,agg):
        (c,k), (r, mux, sigx) = pol, agg
        (ii,jj), p_func = ind, {}
        pup_y = (self.dt/self.Delta_y**2)*(self.SIGSIG**2*k**2/2 + self.Delta_y*(r + (self.Pi-r)*k))
        pdown_y = (self.dt/self.Delta_y**2)*(self.SIGSIG**2*k**2/2 + self.Delta_y*(c + self.SIGSIG**2*k**2/2))
        d = [self.dt/self.Delta[i]**2 for i in range(2)]
        muK, musig = sigx*(1-self.gamma)*self.XX*self.SIGSIG*k, self.sigsig*(1-self.gamma)*self.SIGSIG*k
        pup_x = d[0]*self.Delta[0]*(np.maximum(mux*self.XX,0)+np.maximum(muK,0))
        pdown_x = d[0]*self.Delta[0]*(np.maximum(-mux*self.XX,0)+np.maximum(-muK,0))
        pup_sig = d[1]*self.Delta[1]*(np.maximum(self.theta*(self.sigbar-self.SIGSIG)+np.maximum(musig,0),0))
        pdown_sig = d[1]*self.Delta[1]*(np.maximum(-self.theta*(self.sigbar-self.SIGSIG)+np.maximum(-musig,0),0))
        p_func['up'], p_func['down'] = pup_y[ii,jj], pdown_y[ii,jj]
        p_func[(1,0)], p_func[(-1,0)] = pup_x[ii,jj], pdown_x[ii,jj]
        p_func[(0,1)], p_func[(0,-1)] = pup_sig[ii,jj], pdown_sig[ii,jj]
        return p_func

    #following is E[V(x',sigma')*np.exp((1-gamma)*('y-y))]
    def H(self,type,pol,agg):
        (c, k), rho = pol, self.rho[type]
        (m1,m2), pbar, z = self.NL(agg)
        ii, jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),indexing = 'ij')
        p, e = self.p_func((ii,jj),type,pol,agg), np.exp((1-self.gamma)*self.Delta_y)
        diag = 1 + p['up']*(e-1) + p['down']*(1/e-1) - 2*pbar \
        - p[(1,0)] - p[(-1,0)] - p[(0,1)] - p[(0,-1)]
        H = self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj,diag)
        H = H+self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj+m1[0]*(self.N[1]+1)+m1[1],(1-np.abs(z))*pbar)
        H = H+self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj-m1[0]*(self.N[1]+1)-m1[1],(1-np.abs(z))*pbar)
        H = H+self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj+m2[0]*(self.N[1]+1)+m2[1],np.abs(z)*pbar)
        H = H+self.T_func(ii*(self.N[1]+1)+jj,ii*(self.N[1]+1)+jj-m2[0]*(self.N[1]+1)-m2[1],np.abs(z)*pbar)
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)),indexing='ij')
            H = H + self.T_func(ii*(self.N[1]+1)+jj,(ii+key[0])*(self.N[1]+1)+jj+key[1],self.p_func((ii,jj),type,pol,agg)[key])
        return H

    #following creates \overline{T} operator. Explicit expressions are in appendix.
    def T(self,type,pol,agg):
        return (1/self.dt)*(np.exp(-self.rho[type]*self.dt)*self.H(type,pol,agg) - sp.eye(self.M))/(1-self.gamma)

    #following creates non-local part of transition probabilities
    #return the integer transitions, transition probabilities, and weights between
    #the two pairs.
    def NL(self,agg):
        m1 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        m2 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        sig_grid = agg[2]*self.XX/self.Delta[0], self.sigsig/self.Delta[1] # + 0*self.SIGSIG
        with np.errstate(divide='ignore',invalid='ignore'):
            w = sig_grid[0]/sig_grid[1]
            val = [(np.minimum(1,w)*np.abs(j-i/w)).reshape(self.M,) for i,j in self.index]
            I = np.argmin(np.array(val),axis=0).reshape((self.N[0]+1,self.N[1]+1))
            for n in range(len(self.index)):
                m1[0][I==n],m1[1][I==n] = self.index[n]
                z = np.minimum(1,w)*(m1[1]-m1[0]/w)
            m2 = m1[0] + (w<=1)*(2*(z>0) - 1), m1[1] + (w>1)*(2*(z<=0) - 1)
            #No second point on boundary:
            z[:,0], z[:,-1], z[0,:], z[-1,:] = 0, 0, 0, 0
            m2[0][:,0], m2[0][:,-1], m2[0][0,:], m2[0][-1,:] = 0, 0, 0, 0
            m2[1][:,0], m2[1][:,-1], m2[1][0,:], m2[1][-1,:] = 0, 0, 0, 0
            #adjust on boundary:
            m1[0][0,:], m1[0][-1,:], m1[1][0,:], m1[1][-1,:] = 0, 0, 1, 1
            m1[0][:,0], m1[0][:,-1], m1[1][:,0], m1[1][:,-1] = 1, 1, 0, 0
            Dt_bar = (w>1)*2*m1[0]**2/sig_grid[0]**2 + (w<=1)*2*m1[1]**2/sig_grid[1]**2
            Dt_bar[0,:], Dt_bar[-1,:] = 2/sig_grid[1][0,:]**2, 2/sig_grid[1][-1,:]**2
            Dt_bar[:,0], Dt_bar[:,-1] = 2/sig_grid[0][:,0]**2, 2/sig_grid[0][:,-1]**2
        (m1,m2) = (self.bound_adj(m1),self.bound_adj(m2))
        #adjust on transitions at corners where both vanish
        m1[0][0,0], m1[0][-1,0], m1[0][0,-1], m1[0][-1,-1] = 0, 0, 0, 0
        m1[1][0,0], m1[1][-1,0], m1[1][0,-1], m1[1][-1,-1] = 0, 0, 0, 0
        Dt_bar[0,0], Dt_bar[-1,0], Dt_bar[0,-1], Dt_bar[-1,-1] = 1, 1, 1, 1
        return (m1,m2), self.dt/Dt_bar, z

    #optimal policy for each type, given continuation values, typo and law of motion
    #of aggregate state. V is single array not pairs of arrays. agg = (r, mux, sigx)
    def polupdate(self,type,V,agg):
        E1, E2, Ec = self.con_E()
        VEFx, VEFsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEBx, VEBsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEFx[:-1,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        VEFsig[:,:-1] = (V[:,1:] - V[:,:-1])/self.Delta[1]
        VEBx[1:,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        VEBsig[:,1:] = (V[:,1:] - V[:,:-1])/self.Delta[1]
        Vpx = (self.gamma<1)*VEFx + (self.gamma>1)*VEBx
        Vpsig = (self.gamma<1)*VEFsig + (self.gamma>1)*VEBsig
        k = np.maximum((E1/E2)*(self.Pi - agg[0])/(self.gamma*self.SIGSIG**2) \
        + E2**(-1)*(agg[2]*self.XX*Vpx + self.sigsig*Vpsig)/(self.gamma*self.SIGSIG*V), 0)
        d = np.exp(self.rho[type]*self.dt/self.gamma)*Ec
        return (self.rho[type]/V)**(1/self.gamma)*d, (type==0)*k + (type==1)*0

    #update value function using PFI, given POLICY function
    def Vupdate_PFI(self,type,pol,agg):
        b = self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        return sp.linalg.spsolve(-self.T(type,pol,agg), b.reshape((self.M,1))).reshape((self.N[0]+1,self.N[1]+1))

    #update value function using PFI, given PREVIOUS VALUE function
    def updateV_PFI(self,type,V,agg):
        return self.Vupdate_PFI(type,self.polupdate(type,V,agg),agg)

    #solve individual problem using PFI given aggregate quantities
    def solveV_PFI(self,type,agg):
        c,k = self.rho[type]/10 + 0*agg[0], 0*agg[0]
        V = self.Vupdate_PFI(type,(c,k),agg)
        (c,k) = self.polupdate(type,V,agg)
        check_NaN = np.isnan(agg[0]).any()+np.isnan(agg[1]).any()+np.isnan(agg[2]).any()
        eps, eps2, i = 1, 1, 1
        if check_NaN==1:
            print("ERROR: NaNs in equilibrium quantities")
        while i < 40 and eps > self.tol:
            p = self.p_func((self.ii_,self.jj_),type,(c,k),agg)
            probs = sum(p.values()) + 2*self.NL(agg)[1]
            if np.max(probs) > 1:
                print("ERROR: probabilities negative")
            V1 = self.Vupdate_PFI(type,(c,k),agg)
            (c1,k1) = self.polupdate(type,V1,agg)
            eps = np.max(np.abs(V**(1/(1-self.gamma)) - V1**(1/(1-self.gamma))))
            eps2 = np.max(np.abs(c - c1)) + np.max(np.abs(k - k1))
            if np.min(V1/(1-self.gamma)-V/(1-self.gamma)) < -10**(-4):
                diff = V1/(1-self.gamma)-V/(1-self.gamma)
                print("Failure of monotonicity at:", len(V[diff<-10**(-4)]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean(diff[diff<-10**(-4)]))
            if np.min(V1) < 0:
                print("ERROR: value function becomes negative")
            V, (c,k), i = V1, (c1,k1), i+1
        if eps > self.tol or np.isnan(V).any() or np.min(V1)<0:
            print("Individual problem did not converge", "Difference:", eps)
        else:
            print("Convergence of individual problem in", i, "iterations.", "Difference:", eps)
        return V

    def Vupdate(self,type,pol,V,M,agg):
        b = self.dt*self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        V, T = V.reshape((self.M,1)), self.T(type,pol,agg)
        for i in range(M+1):
            V = V + (1-self.gamma)*(b.reshape((self.M,1)) + self.dt*T*V)
        return V.reshape((self.N[0]+1,self.N[1]+1))

    def updateV(self,type,V,agg,M):
        return self.Vupdate(type,self.polupdate(type,V,agg),V,M,agg)

    def agg_update(self,V_pair):
        (E1, E2, Ec), VE = self.con_E(), V_pair[0]
        VEFx, VEFsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEBx, VEBsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEFx[:-1,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        VEFsig[:,:-1] = (VE[:,1:] - VE[:,:-1])/self.Delta[1]
        VEBx[1:,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        VEBsig[:,1:] = (VE[:,1:] - VE[:,:-1])/self.Delta[1]
        Vpx = (self.gamma<1)*VEFx + (self.gamma>1)*VEBx
        Vpsig = (self.gamma<1)*VEFsig + (self.gamma>1)*VEBsig
        denom = self.gamma + self.XX*(1-self.XX)*(-Vpx)/(E2*VE)
        with np.errstate(divide='ignore',invalid='ignore'):
            sigx = self.SIGSIG*(1-self.XX)*np.minimum(((E1/E2)*(self.Pi-self.rlow)/self.SIGSIG**2 \
            + self.sigsig*Vpsig/(self.SIGSIG*E2*VE))/denom, 1/self.XX)
            r = np.maximum(self.rlow,self.Pi + E1**(-1)*self.SIGSIG*(sigx*self.XX*Vpx+self.sigsig*Vpsig)/VE \
            -(E1/E2)**(-1)*self.gamma*self.SIGSIG**2/self.XX)
        r[self.XX==0], sigx[self.XX==0] = self.rlow, 0
        k = np.maximum((E1/E2)*(self.Pi - r)/(self.gamma*self.SIGSIG**2) \
        + (sigx*self.XX*Vpx+self.sigsig*Vpsig)/(self.gamma*self.SIGSIG*VE*E2), 0)
        c = (self.rho[0]/V_pair[0])**(1/self.gamma)*np.exp(self.rho[0]*self.dt/self.gamma)*Ec, \
        (self.rho[1]/V_pair[1])**(1/self.gamma)*np.exp(self.rho[1]*self.dt/self.gamma)*Ec
        mux = (c[1] - c[0] + (self.Pi-r)*k - self.SIGSIG**2*k**2*self.XX)*(1-self.XX)
        return r, mux, sigx

    #solve for the competitive equilibrium using policy iteration approach
    def solve_PFI(self):
        print("Solving with policy iteration approach")
        tic, j, i = time.time(), 0, 0
        agg, eps = self.log_quant(), 1
        V_pair = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
        while j < len(self.relax):
            print("Trying level of relaxation:", j)
            while i < self.pol_maxiter and eps > self.tol:
                r, mux, sigx = self.agg_update(V_pair)
                DIFF = agg[0] - r, self.XX*(agg[1] - mux), self.XX*(agg[2] - sigx)
                eps = np.amax(np.abs(DIFF[0][:,1:-1])) + np.amax(np.abs(DIFF[1][:,1:-1])) \
                + np.amax(np.abs(DIFF[2][:,1:-1]))
                agg2 = r*(1-self.relax[j]) + self.relax[j]*agg[0], mux*(1-self.relax[j]) \
                + self.relax[j]*agg[1], sigx*(1-self.relax[j]) + self.relax[j]*agg[2]
                V_pair2 = (self.solveV_PFI(0,agg2), self.solveV_PFI(1,agg2))
                V_pair, agg, i = V_pair2, agg2, i+1
                print("Outer iteration in PFI:", i, "Differences:", eps)
            if eps < self.tol:
                print("PFI converged in:", i, "iterations")
                print("with relaxation level:", j)
                toc = time.time()
                return (r, mux, sigx), toc-tic
            else:
                print("PFI did not converge after:", i, "iterations")
                print("with relaxation level:", j, "Difference:", eps)
                i, j= 0, j+1
        toc = time.time()
        return (r, mux, sigx), toc-tic

    #solve for the competitive equilibrium using the false transient approach
    def solve_FT(self,check):
        print("Solving with false transient approach")
        agg_PFI = self.solve_PFI()[0]
        if check==1:
            agg = agg_PFI
            V_pair = self.solveV_PFI(0,agg), self.solveV_PFI(1,agg)
        else:
            agg = self.log_quant()
            V_pair = 1+0*self.solveV_PFI(0,agg), 1+0*self.solveV_PFI(1,agg)
        eps, eps2, i = 1, 1, 0
        diff = np.amax(np.abs(agg[0] - agg_PFI[0])) + np.amax(np.abs((agg[1] - agg_PFI[1])*self.XX)) \
         + np.amax(np.abs((agg[2] - agg_PFI[2])*self.XX))
        difference, iterations = [diff], [0]
        tic = time.time()
        while i < self.max_iter_eq and eps > 0*self.tol:
            r, mux, sigx = self.agg_update(V_pair)
            DIFF = r - agg[0], self.XX*(mux - agg[1]), self.XX*(sigx - agg[2])
            eps = np.amax(np.abs(DIFF[0][:,1:-1])) + np.amax(np.abs(DIFF[1][:,1:-1])) + np.amax(np.abs(DIFF[2][:,1:-1]))
            eps2 = np.amax(np.abs(r - agg_PFI[0])) + np.amax(np.abs((mux - agg_PFI[1])*self.XX)) + np.amax(np.abs((sigx - agg_PFI[2])*self.XX))
            V_pair2 = (self.updateV(0,V_pair[0],(r, mux, sigx),0), self.updateV(1,V_pair[1],(r, mux, sigx),0))
            V_pair, agg, i = V_pair2, (r, mux, sigx), i+1
            if i % 100 == 0:
                print("Outer iteration:", i, "Differences:", eps,eps2)
                p = self.p_func((self.ii_,self.jj_),0,self.polupdate(0,V_pair[0],agg),agg)
                probs = sum(p.values()) + 2*self.NL(agg)[1]
                print("sum of transition probs:", np.max(probs))
                print("sum of non-y transition probs:", np.max(p[(1,0)]+p[(-1,0)]+p[(0,1)]+p[(0,-1)]))
                difference.append(eps2), iterations.append(i)
                if np.max(probs) > 1:
                    print("ERROR: probabilities negative")
        toc = time.time()
        print("Number of outer iterations:", i)
        print("Time taken for false transient:", toc-tic)
        return (r, mux, sigx), (np.array(difference), np.array(iterations)), toc-tic

    def bound_adj(self,m):
        i, j = np.meshgrid(range(self.N[0]+1), range(self.N[1]+1),indexing='ij')
        return np.minimum(m[0],np.minimum(self.mbar,np.minimum(i,self.N[0]-i))), \
        np.minimum(m[1],np.minimum(self.mbar,np.minimum(j,self.N[1]-j)))

    def grid(self,m):
        x1 = np.linspace(self.X_bnd[0][0]+m*self.Delta[0],self.X_bnd[0][1]-m*self.Delta[0],self.N[0]+1-2*m)
        x2 = np.linspace(self.X_bnd[1][0]+m*self.Delta[1],self.X_bnd[1][1]-m*self.Delta[1],self.N[1]+1-2*m)
        return np.meshgrid(x1,x2,indexing='ij')

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def con_E(self):
        E1 = (np.exp((1-self.gamma)*self.Delta_y) - 1)/(self.Delta_y*(1-self.gamma))
        E2 = ((2 - np.exp(-(1-self.gamma)*self.Delta_y) - np.exp((1-self.gamma)*self.Delta_y))/self.Delta_y**2 \
        + (1 - np.exp(-(1-self.gamma)*self.Delta_y))/self.Delta_y)/(self.gamma*(1-self.gamma))
        Ec = (self.Delta_y*(1-self.gamma)/(1-np.exp(-(1-self.gamma)*self.Delta_y)))**(1/self.gamma)
        return E1, E2, Ec

    def log_quant(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            r_log = np.maximum(self.rlow, self.Pi - self.SIGSIG**2/self.XX)
            kbar = np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
            sigx_log = self.SIGSIG*(1-self.XX)*np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
        r_log[self.XX==0], sigx_log[self.XX==0], kbar[self.XX==0] = self.rlow, 0, 0
        mux_log = (self.rho[1] - self.rho[0] + (self.Pi-r_log)*kbar - self.SIGSIG**2*kbar**2*self.XX)*(1-self.XX)
        return r_log, mux_log, sigx_log

    def V_bnd(self,type,r):
        s = (type==0)*self.con_E()[0]*(self.Pi-r)**2/(2*self.con_E()[1]*self.SIGSIG**2*self.gamma)
        denom = self.rho[type] - (1-self.gamma)*self.con_E()[0]*(r + s)
        if np.min(denom) < 0:
            print("Boundary values not well-defined")
        else:
            return self.gamma**self.gamma*self.rho[type] \
            *self.con_E()[2]**(self.gamma*(1-self.gamma))*denom**(-self.gamma)

    def r_bnd(self,x):
        if x==0:
            return self.rlow
        else:
            return np.maximum(self.rlow,self.Pi - self.gamma*self.SIGSIG**2*(self.con_E()[0]/self.con_E()[1])**(-1))

    def V_bnd_eq(self,type,x):
        return self.V_bnd(type,self.r_bnd(x))

    def V_bnd_lin(self,type):
        return self.V_bnd_eq(type,0)*(1-self.XX) + self.V_bnd_eq(type,1)*self.XX

"""
Correlated noise with state-dependent timestep
"""

class MF_corr_var_dt(object):
    def __init__(self,rho=[.2, .1],gamma=0.9995,Pi=0.065,rlow=0.0, sigsigbar=0.2,
    theta=1, N=(200,50), X_bnd=[[0,1],[0.1,0.4]],mbar=4,tol=10**-6, pbar = 0.01,
    Delta_y=0.01, max_iter_eq=100,pol_maxiter = 20, relax=0.0):
        self.rho, self.gamma, self.Pi, self.rlow = rho, gamma, Pi, rlow
        self.theta, self.sigsigbar, self.X_bnd = theta, sigsigbar, X_bnd
        self.sigbar = (self.X_bnd[1][1] + self.X_bnd[1][0])/2
        self.N, self.M = N, (N[0]+1)*(N[1]+1)
        self.mbar, self.tol = mbar, tol
        self.pbar, self.Delta_y = pbar, Delta_y
        self.max_iter_eq = max_iter_eq
        self.Delta = (self.X_bnd[0][1]-self.X_bnd[0][0])/self.N[0], (self.X_bnd[1][1]-self.X_bnd[1][0])/self.N[1]
        self.index = [(i,j) for i,j in itertools.product(range(self.mbar),range(self.mbar)) if (i,j)!=(0,0)]
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.XX, self.SIGSIG = self.grid(0)
        self.sigsig = self.sigsigbar*(self.SIGSIG<self.X_bnd[1][1] - 0*self.Delta[1])*(self.SIGSIG>self.X_bnd[1][0] + 0*self.Delta[1])
        self.relax, self.pol_maxiter = relax, pol_maxiter

    def tran_func(self,ind,pol,agg):
        (c,k), (r, mux, sigx) = pol, agg
        (ii,jj), tran_func = ind, {}
        muK, musig = sigx*(1-self.gamma)*self.XX*self.SIGSIG*k, self.sigsig*(1-self.gamma)*self.SIGSIG*k
        tran_func[(1,0)] = ((np.maximum(mux*self.XX,0)+np.maximum(muK,0))/self.Delta[0])[ii,jj]
        tran_func[(-1,0)] = ((np.maximum(-mux*self.XX,0)+np.maximum(-muK,0))/self.Delta[0])[ii,jj]
        tran_func[(0,1)] = ((np.maximum(self.theta*(self.sigbar-self.SIGSIG),0)+np.maximum(musig,0))/self.Delta[1])[ii,jj]
        tran_func[(0,-1)] = ((np.maximum(-self.theta*(self.sigbar-self.SIGSIG),0)+np.maximum(-musig,0))/self.Delta[1])[ii,jj]
        return tran_func

    def T(self,type,pol,agg):
        (m1,m2), Dt_bar, z = self.NL(agg)
        ii, jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),indexing='ij')
        row = ii*(self.N[1]+1)+jj
        (c, k), rho = pol, (type==0)*self.rho[0] + (type==1)*self.rho[1]
        discount = np.exp(-rho*self.pbar*Dt_bar)
        E1, E2, Ec = self.con_E()
        d_lead = (discount[ii,jj] - 1)/(self.pbar*Dt_bar[ii,jj]) \
        + discount[ii,jj]*(1-self.gamma)*(E1*(agg[0]-c*np.exp(-(1-self.gamma)*self.Delta_y) \
        + (self.Pi-agg[0])*k) - E2*self.gamma*self.SIGSIG**2*k**2/2) \
        - 2*discount[ii,jj]/Dt_bar - discount[ii,jj]*sum(self.tran_func((ii,jj),pol,agg).values())
        T = self.T_func(row,row,d_lead)
        #non-local transition coefficients:
        T = T+self.T_func(row,row+m1[0]*(self.N[1]+1)+m1[1],discount[ii,jj]*(1-np.abs(z))/Dt_bar)
        T = T+self.T_func(row,row-m1[0]*(self.N[1]+1)-m1[1],discount[ii,jj]*(1-np.abs(z))/Dt_bar)
        T = T+self.T_func(row,row+m2[0]*(self.N[1]+1)+m2[1],discount[ii,jj]*np.abs(z)/Dt_bar)
        T = T+self.T_func(row,row-m2[0]*(self.N[1]+1)-m2[1],discount[ii,jj]*np.abs(z)/Dt_bar)
        #local transition coefficients:
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)),indexing='ij')
            row, col = ii*(self.N[1]+1)+jj, (ii+key[0])*(self.N[1]+1)+jj+key[1]
            prob = discount[ii,jj]*self.tran_func((ii,jj),pol,agg)[key]
            T = T+self.T_func(row,col,prob)
        #DON'T FORGET TO NORMALIZE:
        return T/(1-self.gamma)

    def NL(self,agg):
        m1 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        m2 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        sig_grid = agg[2]*self.XX/self.Delta[0], self.sigsig/self.Delta[1] # + 0*self.SIGSIG
        with np.errstate(divide='ignore',invalid='ignore'):
            w = sig_grid[0]/sig_grid[1]
            val = [(np.minimum(1,w)*np.abs(j-i/w)).reshape(self.M,) for i,j in self.index]
            I = np.argmin(np.array(val),axis=0).reshape((self.N[0]+1,self.N[1]+1))
            for n in range(len(self.index)):
                m1[0][I==n],m1[1][I==n] = self.index[n]
                z = np.minimum(1,w)*(m1[1]-m1[0]/w)
            m2 = m1[0] + (w<=1)*(2*(z>0) - 1), m1[1] + (w>1)*(2*(z<=0) - 1)
            #No second point on boundary:
            z[:,0], z[:,-1], z[0,:], z[-1,:] = 0, 0, 0, 0
            m2[0][:,0], m2[0][:,-1], m2[0][0,:], m2[0][-1,:] = 0, 0, 0, 0
            m2[1][:,0], m2[1][:,-1], m2[1][0,:], m2[1][-1,:] = 0, 0, 0, 0
            #Adjust so that we run ALONG edge. First x on bnd, then sig on bnd.
            m1[0][0,:], m1[0][-1,:], m1[1][0,:], m1[1][-1,:] = 0, 0, 1, 1
            m1[0][:,0], m1[0][:,-1], m1[1][:,0], m1[1][:,-1] = 1, 1, 0, 0
            Dt_bar = (w>1)*2*m1[0]**2/sig_grid[0]**2 + (w<=1)*2*m1[1]**2/sig_grid[1]**2
            Dt_bar[0,:], Dt_bar[-1,:] = 2/sig_grid[1][0,:]**2, 2/sig_grid[1][-1,:]**2
            Dt_bar[:,0], Dt_bar[:,-1] = 2/sig_grid[0][:,0]**2, 2/sig_grid[0][:,-1]**2
        (m1,m2) = (self.bound_adj(m1),self.bound_adj(m2))
        #adjust on transitions at corners where both vanish
        m1[0][0,0], m1[0][-1,0], m1[0][0,-1], m1[0][-1,-1] = 0, 0, 0, 0
        m1[1][0,0], m1[1][-1,0], m1[1][0,-1], m1[1][-1,-1] = 0, 0, 0, 0
        Dt_bar[0,0], Dt_bar[-1,0], Dt_bar[0,-1], Dt_bar[-1,-1] = 1, 1, 1, 1
        return (m1,m2), Dt_bar, z

    def Dt_bar(self,sigx):
        m1 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        m2 = (np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1)))
        sig_grid = sigx*self.XX/self.Delta[0], self.sigsig/self.Delta[1] + 0*self.SIGSIG
        with np.errstate(divide='ignore',invalid='ignore'):
            w = sig_grid[0]/sig_grid[1]
            val = [(np.minimum(1,w)*np.abs(j-i/w)).reshape(self.M,) for i,j in self.index]
            I = np.argmin(np.array(val),axis=0).reshape((self.N[0]+1,self.N[1]+1))
            for n in range(len(self.index)):
                m1[0][I==n],m1[1][I==n] = self.index[n]
                z = np.minimum(1,w)*(m1[1]-m1[0]/w)
            m2 = m1[0] + (w<=1)*(2*(z>0) - 1), m1[1] + (w>1)*(2*(z<=0) - 1)
            #No second point on boundary:
            z[:,0], z[:,-1], z[0,:], z[-1,:] = 0, 0, 0, 0
            m2[0][:,0], m2[0][:,-1], m2[0][0,:], m2[0][-1,:] = 0, 0, 0, 0
            m2[1][:,0], m2[1][:,-1], m2[1][0,:], m2[1][-1,:] = 0, 0, 0, 0
            #Adjust so that we run ALONG edge. First x on bnd, then sig on bnd.
            m1[0][0,:], m1[0][-1,:], m1[1][0,:], m1[1][-1,:] = 0, 0, 1, 1
            m1[0][:,0], m1[0][:,-1], m1[1][:,0], m1[1][:,-1] = 1, 1, 0, 0
            Dt_bar = (w>1)*2*m1[0]**2/sig_grid[0]**2 + (w<=1)*2*m1[1]**2/sig_grid[1]**2
            Dt_bar[0,:], Dt_bar[-1,:] = 2/sig_grid[1][0,:]**2, 2/sig_grid[1][-1,:]**2
            Dt_bar[:,0], Dt_bar[:,-1] = 2/sig_grid[0][:,0]**2, 2/sig_grid[0][:,-1]**2
        (m1,m2) = (self.bound_adj(m1),self.bound_adj(m2))
        #adjust on transitions at corners where both vanish
        m1[0][0,0], m1[0][-1,0], m1[0][0,-1], m1[0][-1,-1] = 0, 0, 0, 0
        m1[1][0,0], m1[1][-1,0], m1[1][0,-1], m1[1][-1,-1] = 0, 0, 0, 0
        Dt_bar[0,0], Dt_bar[-1,0], Dt_bar[0,-1], Dt_bar[-1,-1] = 1,1,1,1
        return Dt_bar

    def polupdate(self,type,V,agg):
        E1, E2, Ec = self.con_E()
        VEFx, VEFsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEBx, VEBsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEFx[:-1,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        VEFsig[:,:-1] = (V[:,1:] - V[:,:-1])/self.Delta[1]
        VEBx[1:,:] = (V[1:,:] - V[:-1,:])/self.Delta[0]
        VEBsig[:,1:] = (V[:,1:] - V[:,:-1])/self.Delta[1]
        Vpx = (self.gamma<1)*VEFx + (self.gamma>1)*VEBx
        Vpsig = (self.gamma<1)*VEFsig + (self.gamma>1)*VEBsig
        k = np.maximum((E1/E2)*(self.Pi - agg[0])/(self.gamma*self.SIGSIG**2) \
        + E2**(-1)*(agg[2]*self.XX*Vpx + self.sigsig*Vpsig)/(self.gamma*self.SIGSIG*V), 0)
        Delta_t = self.pbar*self.Dt_bar(agg[2])
        d = np.exp(self.rho[type]*Delta_t/self.gamma)*Ec
        return (self.rho[type]/V)**(1/self.gamma)*d, (type==0)*k + (type==1)*0

    def Vupdate_PFI(self,type,pol,agg):
        b = self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        return sp.linalg.spsolve(-self.T(type,pol,agg), b.reshape((self.M,1))).reshape((self.N[0]+1,self.N[1]+1))

    def updateV_PFI(self,type,V,agg):
        return self.Vupdate_PFI(type,self.polupdate(type,V,agg),agg)

    def solveV_PFI(self,type,agg):
        c,k = self.rho[type]/10 + 0*agg[0], 0*agg[0]
        V = self.Vupdate_PFI(type,(c,k),agg)
        (c,k) = self.polupdate(type,V,agg)
        check_NaN = np.isnan(agg[0]).any()+np.isnan(agg[1]).any()+np.isnan(agg[2]).any()
        eps, eps2, i = 1, 1, 1
        if check_NaN==1:
            print("ERROR: NaNs in equilibrium quantities")
        while i < 20 and eps2 > self.tol:
            V1 = self.Vupdate_PFI(type,(c,k),agg)
            (c1,k1) = self.polupdate(type,V1,agg)
            eps = np.mean(np.abs(V**(1/(1-self.gamma)) - V1**(1/(1-self.gamma))))
            eps2 = np.max(np.abs(c - c1)) + np.max(np.abs(k - k1))
            if np.min(V1) < 0:
                print("ERROR: value function becomes negative")
            V, (c,k), i = V1, (c1,k1), i+1
        if eps > self.tol or np.isnan(V).any() or np.min(V1)<0:
            print("Individual problem did not converge")
        else:
            print("Convergence of individual problem in", i, "iterations")
        return V

    def Vupdate(self,type,pol,V,M,agg):
        Delta_t = self.pbar*self.Dt_bar(agg[2])
        b = Delta_t*self.rho[type]*pol[0]**(1-self.gamma)/(1-self.gamma)
        V, T = V.reshape((self.M,1)), self.T(type,pol,agg)
        Dt = Delta_t.reshape((self.M,))
        for i in range(M+1):
            V = V + (1-self.gamma)*(b.reshape((self.M,1)) + sp.diags(Dt)*T*V)
        return V.reshape((self.N[0]+1,self.N[1]+1))

    def updateV(self,type,V,agg,M):
        return self.Vupdate(type,self.polupdate(type,V,agg),V,M,agg)

    def agg_update(self,V_pair):
        (E1, E2, Ec), VE = self.con_E(), V_pair[0]
        VEFx, VEFsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEBx, VEBsig = np.zeros((self.N[0]+1,self.N[1]+1)), np.zeros((self.N[0]+1,self.N[1]+1))
        VEFx[:-1,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        VEFsig[:,:-1] = (VE[:,1:] - VE[:,:-1])/self.Delta[1]
        VEBx[1:,:] = (VE[1:,:] - VE[:-1,:])/self.Delta[0]
        VEBsig[:,1:] = (VE[:,1:] - VE[:,:-1])/self.Delta[1]
        Vpx = (self.gamma<1)*VEFx + (self.gamma>1)*VEBx
        Vpsig = (self.gamma<1)*VEFsig + (self.gamma>1)*VEBsig
        denom = self.gamma + self.XX*(1-self.XX)*(-Vpx)/(E2*VE)
        with np.errstate(divide='ignore',invalid='ignore'):
            sigx = self.SIGSIG*(1-self.XX)*np.minimum(((E1/E2)*(self.Pi -self.rlow)/self.SIGSIG**2 \
            + self.sigsig*Vpsig/(self.SIGSIG*E2*VE))/denom, 1/self.XX)
            r = np.maximum(self.rlow,self.Pi + E1**(-1)*self.SIGSIG*(sigx*self.XX*Vpx+self.sigsig*Vpsig)/VE \
            -(E1/E2)**(-1)*self.gamma*self.SIGSIG**2/self.XX)
        r[self.XX==0], sigx[self.XX==0] = self.rlow, 0
        k = np.maximum((E1/E2)*(self.Pi - r)/(self.gamma*self.SIGSIG**2) \
        + (sigx*self.XX*Vpx+self.sigsig*Vpsig)/(self.gamma*self.SIGSIG*VE*E2), 0)
        Delta_t = self.pbar*self.Dt_bar(sigx)
        c = (self.rho[0]/V_pair[0])**(1/self.gamma)*np.exp(self.rho[0]*Delta_t/self.gamma)*Ec, \
        (self.rho[1]/V_pair[1])**(1/self.gamma)*np.exp(self.rho[1]*Delta_t/self.gamma)*Ec
        mux = (c[1] - c[0] + (self.Pi-r)*k - self.SIGSIG**2*k**2*self.XX)*(1-self.XX)
        return r, mux, sigx

    def solve_PFI(self):
        agg, eps, i = self.log_quant(), 1, 1
        V_pair = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
        while i < self.pol_maxiter and eps > self.tol and eps < 100:
            r, mux, sigx = self.agg_update(V_pair)
            eps = np.amax(np.abs(r - agg[0])) + np.amax(np.abs(mux - agg[1])) + np.amax(np.abs(sigx - agg[2]))
            agg = r*(1-self.relax) + self.relax*agg[0], mux*(1-self.relax) \
            + self.relax*agg[1], sigx*(1-self.relax) + self.relax*agg[2]
            V_pair2 = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
            eps_V = np.amax(np.abs(V_pair2[0] - V_pair[0])) + np.amax(np.abs(V_pair2[1] - V_pair[1]))
            V_pair, i = V_pair2, i+1
            print("Outer iteration in PFI:", i, "Difference:", eps)
        if eps < self.tol:
            print("PFI converged in:", i, "iterations")
            print("Difference:", eps)
        else:
            print("PFI did not converge after:", i, "iterations")
            print("Difference:", eps)
        return r, mux, sigx

    def solve_PFI_alt(self):
        r, mux, sigx = 0*self.XX, 0*self.XX, self.SIGSIG*(1-self.XX)
        agg = (r, mux, sigx)
        eps,i = 1, 1
        while i < 20 and eps > self.tol:
            V = (self.solveV_PFI(0,agg), self.solveV_PFI(1,agg))
            r, mux, sigx = self.agg_update(V)
            eps = np.mean(np.abs(r - agg[0])) + np.mean(np.abs(mux - agg[1])) + np.mean(np.abs(sigx - agg[2]))
            print(eps, np.mean(np.abs(mux - agg[1])), np.mean(np.abs(sigx - agg[2])))
            agg = (r, mux, sigx)
            i = i+1
        return r,mux,sigx

    def solve_FT(self,check):
        agg_PFI = self.solve_PFI()
        if check==1:
            agg = agg_PFI
            V_pair = self.solveV_PFI(0,agg), self.solveV_PFI(1,agg)
        else:
            agg = self.log_quant()
            V_pair = self.solveV_PFI(0,agg), self.solveV_PFI(1,agg)
        eps, eps2, i = 1, 1, 1
        tic = time.time()
        while i < self.max_iter_eq and eps > 0*self.tol:
            r, mux, sigx = self.agg_update(V_pair)
            eps = np.amax(np.abs(r - agg[0])) + np.amax(np.abs((mux - agg[1])*self.XX)) + np.amax(np.abs((sigx - agg[2])*self.XX))
            eps2 = np.amax(np.abs(r - agg_PFI[0])) + np.amax(np.abs((mux - agg_PFI[1])*self.XX)) + np.amax(np.abs((sigx - agg_PFI[2])*self.XX))
            V_pair2 = (self.updateV(0,V_pair[0],(r, mux, sigx),0), self.updateV(1,V_pair[1],(r, mux, sigx),0))
            V_pair, agg, i = V_pair2, (r, mux, sigx), i+1
            if i % 10 == 0:
                print("Outer iteration:", i, "Differences:", eps,eps2)
        toc = time.time()
        print("Number of outer iterations:", i)
        print("Time taken for false transient:", toc-tic)
        return r, mux, sigx

    def bound_adj(self,m):
        i, j = np.meshgrid(range(self.N[0]+1), range(self.N[1]+1),indexing='ij')
        return np.minimum(m[0],np.minimum(self.mbar,np.minimum(i,self.N[0]-i))), \
        np.minimum(m[1],np.minimum(self.mbar,np.minimum(j,self.N[1]-j)))

    def grid(self,m):
        x1 = np.linspace(self.X_bnd[0][0]+m*self.Delta[0],self.X_bnd[0][1]-m*self.Delta[0],self.N[0]+1-2*m)
        x2 = np.linspace(self.X_bnd[1][0]+m*self.Delta[1],self.X_bnd[1][1]-m*self.Delta[1],self.N[1]+1-2*m)
        return np.meshgrid(x1,x2,indexing='ij')

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

    def con_E(self):
        E1 = (np.exp((1-self.gamma)*self.Delta_y) - 1)/(self.Delta_y*(1-self.gamma))
        E2 = ((2 - np.exp(-(1-self.gamma)*self.Delta_y) - np.exp((1-self.gamma)*self.Delta_y))/self.Delta_y**2 \
        + (1 - np.exp(-(1-self.gamma)*self.Delta_y))/self.Delta_y)/(self.gamma*(1-self.gamma))
        Ec = (self.Delta_y*(1-self.gamma)/(1-np.exp(-(1-self.gamma)*self.Delta_y)))**(1/self.gamma)
        return E1, E2, Ec

    def log_quant(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            r_log = np.maximum(self.rlow, self.Pi - self.SIGSIG**2/self.XX)
            kbar = np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
            sigx_log = self.SIGSIG*(1-self.XX)*np.minimum((self.Pi - self.rlow)/self.SIGSIG**2, 1/self.XX)
        r_log[self.XX==0], sigx_log[self.XX==0], kbar[self.XX==0] = self.rlow, 0, 0
        mux_log = (self.rho[1] - self.rho[0] + (self.Pi-r_log)*kbar - self.SIGSIG**2*kbar**2*self.XX)*(1-self.XX)
        return r_log, mux_log, sigx_log

    def V_bnd(self,type,r):
        s = (type==0)*self.con_E()[0]*(self.Pi-r)**2/(2*self.con_E()[1]*self.SIGSIG**2*self.gamma)
        denom = self.rho[type] - (1-self.gamma)*self.con_E()[0]*(r + s)
        if np.min(denom) < 0:
            print("Boundary values not well-defined")
        else:
            return self.gamma**self.gamma*self.rho[type] \
            *self.con_E()[2]**(self.gamma*(1-self.gamma))*denom**(-self.gamma)

    def r_bnd(self,x):
        if x==0:
            return self.rlow
        else:
            return np.maximum(self.rlow,self.Pi - self.gamma*self.SIGSIG**2*(self.con_E()[0]/self.con_E()[1])**(-1))

    def V_bnd_eq(self,type,x):
        return self.V_bnd(type,self.r_bnd(x))

    def V_bnd_lin(self,type):
        return self.V_bnd_eq(type,0)*(1-self.XX) + self.V_bnd_eq(type,1)*self.XX
