"""
Drawing pictures of transitions in case of perfectly correlated diffusions.
Recall w = sigbar[0]/sigbar[1] = (Delta[1]/Delta[0])*(X[0]/X[1]).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time, math, itertools, timeit

sigrat1 = 0.4
sigrat2 = 2.6

x = np.array([0,1,2,3,4])
y = np.array([0,1,2,3,4])

X,Y = np.meshgrid(x,y)
positions = np.vstack([Y.ravel(), X.ravel()])
plt.scatter(*positions[::-1],s=3,c='k')

my_xticks = ['$x_1-\Delta_1$','$x_1$','$x_1+\Delta_1$','$x_1+2\Delta_1$','$x_1+3\Delta_1$']
my_yticks = ['$x_2-\Delta_2$','$x_2$','$x_2+\Delta_2$','$x_2+2\Delta_2$','$x_2+3\Delta_2$']
plt.xticks(x, my_xticks)
plt.yticks(y, my_yticks)
for i in range(1,4):
    for j in range(1,4):
        if (i,j) != (3,2):
            plt.plot(i, j, 'k',markersize=7.5, marker="v")
#plt.plot(3, 1, 'ko',markersize=2.5)
plt.plot(3, 2, 'k',markersize=7.5,marker="s")
plt.plot(3, 1+sigrat1*2, 'k',markersize=7.5,marker="x")
plt.plot(np.linspace(0,4,100), 1+sigrat1*(np.linspace(0,4,100)-1),'k--',label="$\sigma_2/\sigma_1$")
#plt.legend(loc='upper right')
destin = '../../figures/NL1.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
plt.cla()
plt.clf()

x = np.array([0,1,2,3,4])
y = np.array([0,1,2,3,4,5])

X,Y = np.meshgrid(x,y)
positions = np.vstack([Y.ravel(), X.ravel()])
plt.scatter(*positions[::-1],s=8)

my_xticks = ['$x_1-\Delta_1$','$x_1$','$x_1+\Delta_1$','$x_1+2\Delta_1$','$x_1+3\Delta_1$']
my_yticks = ['$x_2-\Delta_2$','$x_2$','$x_2+\Delta_2$','$x_2+2\Delta_2$','$x_2+3\Delta_2$','$x_2+4\Delta_2$']
plt.xticks(x, my_xticks)
plt.yticks(y, my_yticks)
for i in range(1,4):
    for j in range(1,4):
        plt.plot(i, j, 'ro',markersize=2.5)
plt.plot(1, 3, 'ko',markersize=2.5)
plt.plot(2, 3, 'ko',markersize=2.5)
plt.plot(1+2/sigrat2,3, 'ko',markersize=5)
plt.plot(np.linspace(0,3,100), 1+sigrat2*(np.linspace(0,3,100)-1),'k--',label="$\sigma_2/\sigma_1$")
plt.ylim((-0.25,4+0.25))
#plt.legend(loc='upper right')
destin = '../../figures/NL2.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
