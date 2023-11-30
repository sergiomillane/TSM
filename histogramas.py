import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import pandas as pd
import matplotlib.mlab as ml
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#BA = np.loadtxt('total2.txt')
BA = np.loadtxt('ColiDet')
#BA = np.loadtxt('FinalDet')
X=1*np.array(BA[:,0])
XP=1*np.array(BA[:,1])
Y=1*np.array(BA[:,2])
YP=1*np.array(BA[:,3])
Z=1*np.array(BA[:,4])
ZP=1*np.array(BA[:,5])
R=np.sqrt(X*X+Z*Z)
PT=np.sqrt(XP*XP+YP*YP+ZP*ZP)
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
n_bins=250
axs.hist(X, bins = n_bins,alpha=1, color='mediumseagreen',density=False,label='X',range=[-200,200])
axs.hist(Z, bins = n_bins,alpha=1, color='darkslateblue',density=False,label='Y',range=[-200,200])
plt.legend(loc='upper right')
plt.title('')
plt.xlabel('mm')
scale_factor=1
ymin,ymax=plt.ylim()
plt.ylim(ymin*scale_factor,ymax*scale_factor)
plt.ylabel('Counts')
PT=np.sqrt(XP*XP+YP*YP)
#axs.hist(PT, bins = n_bins,alpha=.7, color='yellow')


fig2, axs2 = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
n_bins=200
axs2.hist(XP, bins = n_bins,alpha=1, color='blue',density=False,label='P$_X$')
axs2.hist(ZP, bins = n_bins,alpha=1, color='orange',density=False,label='P$_Z$')
axs2.hist(PT, bins = n_bins,alpha=0.5, color='green',density=False,label='P$_Y$')
plt.legend(loc='upper right')
plt.xlabel('MeV/c')
plt.ylabel('Counts')
plt.title('Momento X,Y,Z')

# Generate non-symmetric test data


fig, ax1= plt.subplots(ncols=1, sharey=True)
#plt.scatter(X,Z)
# Compute 2d histogram. Note the order of x/y and xedges/yedges
ypmean=-np.mean(YP)
#H, yedges, xedges = np.histogram2d(X,1000*XP/ypmean, bins=200,range= [[-50,50],[-500.50,500.50]])
H, yedges, xedges = np.histogram2d(X,PT, bins=200)
#fig, ax1= plt.subplots(ncols=1, sharey=True)
H = np.rot90(H)
H = np.flipud(H)
# Mask zeros
Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 
# Plot 2D histogram using pcolor
fig2 = plt.figure()
plt.pcolormesh(yedges,xedges,Hmasked)
plt.xlabel('X')
plt.ylabel('P$_X$ ')
plt.title('Espacio fase X vs P$_X$')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')

#ax1.pcolormesh(xedges, yedges, H, cmap='rainbow')
print(ypmean)

plt.show()
