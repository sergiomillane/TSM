import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import pandas as pd
import matplotlib.mlab as ml
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#BA = np.loadtxt('total2.txt')
#BA = np.loadtxt('ColiDet')
BA = np.loadtxt('emitancia')
#BA = np.loadtxt('hazideal/ColiDet')
#BA = np.loadtxt('FinalDet')
X=1*np.array(BA[:,0])
Y=1*np.array(BA[:,1])
Z=1*np.array(BA[:,2])
XP=1*np.array(BA[:,3])
YP=1*np.array(BA[:,4])
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
#ax1.hist(R, bins = n_bins,alpha=1, color='blue',density=False,label='P$_X$')
# Compute 2d histogram. Note the order of x/y and xedges/yedges
ypmean=-np.mean(YP)
H, yedges, xedges = np.histogram2d(X,1000*XP/YP, bins=600,range= [[-40,40],[-40,40]])
#H, yedges, xedges = np.histogram2d(0.001*X,1*XP/ypmean, bins=1000)

#H, yedges, xedges = np.histogram2d(X,Z, bins=200)
#fig, ax1= plt.subplots(ncols=1, sharey=True)
H = np.rot90(H)
H = np.flipud(H)
HMAX=np.max(H)
# Mask zeros
Hmasked = np.ma.masked_where(H<0.05*HMAX,H) # Mask pixels with a value of zero
xmid = 0.5*(xedges[1:] + xedges[:-1])
ymid = 0.5*(yedges[1:] + yedges[:-1])
# Calculate averages
# Creamos una malla de coordenadas para los puntos medios
xv, yv = np.meshgrid(xmid, ymid, indexing='ij')

# Calcula la media ponderada para x e y
mean_x = np.sum(Hmasked * xv) / np.sum(Hmasked)
mean_y = np.sum(Hmasked * yv) / np.sum(Hmasked)
print("Media en x: ",mean_x,np.mean(XP))
print("Media en y: ",mean_y,np.mean(X))
# Desviacin estndar ponderada para x e y
std_x = np.sqrt(np.sum(Hmasked * (xv - mean_x)**2) / np.sum(Hmasked))
std_y = np.sqrt(np.sum(Hmasked * (yv - mean_y)**2) / np.sum(Hmasked))
std_yx = np.sqrt(np.sum(Hmasked * (yv - mean_y)*(xv - mean_x)) / np.sum(Hmasked))

print("Desviacion estandar en x: ",std_x,np.std(XP))
print("Desviacion estandar en y: ",std_y,np.std(X),std_yx)
#print(xmid)
Emit=np.sqrt(std_x*std_y-std_yx*std_yx)
print("La emitancia es ",1*Emit," mm.mrad")
# Plot 2D histogram using pcolor
fig2 = plt.figure()
plt.pcolormesh(yedges,xedges,Hmasked)
plt.xlabel('X')
plt.ylabel('P$_X$ ')
plt.title('Espacio fase X vs P$_X$')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')

#ax1.pcolormesh(xedges, yedges, H, cmap='rainbow')
#print(ypmean)


BB = np.loadtxt('ColiDet')
ZPreal=1*np.array(BB[:,5])
fig, axre= plt.subplots(ncols=1, sharey=True)
axre.hist(ZP, bins = n_bins,alpha=1, color='orange',density=True,label='P$_Z$')
axre.hist(ZPreal, bins = n_bins,alpha=0.5, color='Green',density=True,label='P$_Zreal$')

YPreal=1*np.array(BB[:,3])
fig, axre2= plt.subplots(ncols=1, sharey=True)
axre2.hist(-YP, bins = n_bins,alpha=1, color='orange',density=True,label='P$_Z$',range= [0,6])
axre2.hist(-YPreal, bins = n_bins,alpha=0.5, color='Green',density=True,label='P$_Zreal$',range= [0,6])

#n, bins, patches = plt.hist(YP,bins=100)
#print(n)
#print(bins)




plt.show()
