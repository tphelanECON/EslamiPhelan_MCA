"""
LQ problem with degenerate covariance matrix.
Accuracy of 3-pt, 5-pt and BOZ approximations.
Percent errors from various choices of grid sizes and search sizes.
Perhaps create once more with 'ij' indexing to avoid confusion.
"""

import pandas as pd
import numpy as np
import scipy, timeit, time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import LQ_classes, LQ_degen_classes

data_3, data_5, data_BOZ = [], [], []
N_set = [(50,50), (100,100),(150,150),(200,200)]
m_set = [2,4,6,8,10]

Q=np.eye(2)
pbar = 0.001
x_bnd = [[0,1],[0,1]]
rho = 0.15
sigma = 0.3

for n in N_set:
    d3, d5, dBOZ = {}, {}, {}
    for m in m_set:
        X = LQ_degen_classes.LQ_degen3(Q=Q,sigma=sigma,rho=rho,N=n,pbar=pbar,mbar=m,x_bnd=x_bnd)
        tic = time.time()
        V = X.Vsolve()
        toc = time.time()
        error = 100*(X.CF_fun(X.grid(1))-V)/X.CF_fun(X.grid(1))
        print("3-pt error with gridsize", n, "and search length", m, ":", np.sum(np.abs(error))/X.M)
        tic = time.time()
        X.opt_m()
        toc = time.time()
        d3[r'$\overline{m}$ = ' + '{0}'.format(m)] = np.sum(np.abs(error))/X.M
        Y = LQ_degen_classes.LQ_degen5(Q=Q,sigma=sigma,rho=rho,N=n,pbar=pbar,mbar=m,x_bnd=x_bnd)
        tic = time.time()
        V = Y.Vsolve()
        toc = time.time()
        error2 = 100*(Y.CF_fun(Y.grid(1))-V)/Y.CF_fun(Y.grid(1))
        print("5-pt error with gridsize", n, "and search length", m, ":", np.sum(np.abs(error2))/Y.M)
        d5[r'$\overline{m}$ = ' + '{0}'.format(m)] = np.sum(np.abs(error2))/Y.M
        Z = LQ_degen_classes.BOZ(Q=Q,sigma=sigma,rho=rho,N=n,pbar=pbar,mbar=m,x_bnd=[[0,1],[0,1]])
        tic = time.time()
        V = Z.Vsolve()
        toc = time.time()
        error3 = 100*(Z.CF_fun(Z.grid(1))-V)/Z.CF_fun(Z.grid(1))
        print("BOZ error with gridsize", n, "and search length", m, ":", np.sum(np.abs(error3))/Z.M)
        dBOZ[r'$\overline{m}$ = ' + '{0}'.format(m)] = np.sum(np.abs(error3))/Z.M
    data_3.append(d3)
    data_5.append(d5)
    data_BOZ.append(dBOZ)

"""
3-pt table, 5-pt table and BOZ table
"""

"""
destin = 'C:/Users/d1tmp02/Documents/WPs/FSMC/main/figures/LQ_degen3_table.tex'

df = pd.DataFrame(data=data_3,index=N_set)
cols = df.columns.tolist()
cols = []
for m in m_set:
    cols.append(r'$\overline{m}$ = ' + '{0}'.format(m))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

destin = 'C:/Users/d1tmp02/Documents/WPs/FSMC/main/figures/LQ_degen5_table.tex'

df = pd.DataFrame(data=data_5,index=N_set)
cols = df.columns.tolist()
cols = []
for m in m_set:
    cols.append(r'$\overline{m}$ = ' + '{0}'.format(m))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))

destin = 'C:/Users/d1tmp02/Documents/WPs/FSMC/main/figures/LQ_BOZ_table.tex'

df = pd.DataFrame(data=data_BOZ,index=N_set)
cols = df.columns.tolist()
cols = []
for m in m_set:
    cols.append(r'$\overline{m}$ = ' + '{0}'.format(m))

df = df[cols].round(decimals=3)
df.index.names = ['Grid size']

with open(destin,'w') as tf:
    tf.write(df.to_latex(escape=False,column_format='ccccccc'))
"""

"""
Now test speed and the construction times for various subdivisions.
"""

"""
N_set = [(150,150)] #(50,50), (100,100), (250,250),(300,300)
m_set = [6,8] #[2,4]
for n in N_set:
    d3, d5, dBOZ = {}, {}, {}
    for m in m_set:
        X = LQ_degen_classes.LQ_degen3(Q=Q,sigma=0.3,rho=.15,N=n,pbar=pbar,mbar=m,x_bnd=[[0,1],[0,1]])
        tic = time.time()
        BbX = X.Bb()
        toc = time.time()
        print("3-pt time to make matrices for grid size", n, "and steps", m, ":", toc-tic)
        Y = LQ_degen_classes.LQ_degen5(Q=Q,sigma=0.3,rho=.15,N=n,pbar=pbar,mbar=m,x_bnd=[[0,1],[0,1]])
        tic = time.time()
        BbY = Y.Bb()
        toc = time.time()
        print("5-pt time to make matrices for grid size", n, "and steps", m, ":", toc-tic)
        Z = LQ_degen_classes.BOZ(Q=Q,sigma=0.3,rho=.15,N=n,pbar=pbar,mbar=m,x_bnd=[[0,1],[0,1]])
        tic = time.time()
        BbZ = Z.Bb()
        toc = time.time()
        print("BOZ time to make matrices for grid size", n, "and steps", m, ":", toc-tic)
"""

"""
plt.spy(BbX[0],markersize=.1)
plt.show()

plt.spy(BbY[0],markersize=.1)
plt.show()

plt.spy(BbZ[0],markersize=.1)
plt.show()

plt.spy(diff,markersize=.1)
plt.show()
"""
