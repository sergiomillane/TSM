import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os, gzip
import matplotlib.style
import matplotlib as mpl
#mpl.style.use('classic')
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['font.family'] = 'Times New Roman'

# Convert wake from IW2D to PyHT convention
#DEFINE THE ENERGY AND MAIN PARAMETERS
pi=np.pi      		           # pi
Frf=2998e6
Frf=10128000# RF frequency [Hz]   
clight=299792458
lambdabeam=299792458/2998e6 
Omega=Frf*(2*pi)
coveromega=clight/(2*pi*Frf)# CoverOmega
lambdad=1/coveromega
mo=0.5109989461e6
mo=193.729e9
mc=mo/clight
print ("MC= ",mc)
Ekin=10.0e6
gammar=1+(Ekin)/mo              # Relativistic gamma
betar=np.sqrt(1-1.0/(gammar**2))   # Relativistic beta
bg=betar*gammar

print("momentum ",mc)
print ("BG= %.4f M0 = %03s MeV Betar = %03s KEnergy= %03s Mev" % (bg,mo/1e6,betar,Ekin/1e6))
#filename = 'atstripper542.txt'
filename = 'ColiDet'
#ARREGLO B sirve para colocar offsets
B=[0,0,0,0,0,0,0]
#Aqui cargamos los datos en este ejemplo se salta la primeras 10 lineas para evitar leer las cabezeras
A = np.loadtxt(filename, skiprows=0)

dz=clight*4e-12
dz=1
X=np.array(A[:,0])
PX=np.array(A[:,1])
Y=np.array(A[:,2])
Z=np.array(A[:,4])
PY=np.array(A[:,3])
PZ=np.array(A[:,5])+B[5]
PT=np.sqrt(PX*PX+PY*PY+PZ*PZ)
DP=np.std(PT)
print("AAAAAAAAAAAA")
print(np.mean(PT),DP)


fig1=plt.figure()
ax=fig1.add_subplot(111,title="Y vs Y\'", rasterized=True)
nbinsx = 200
limits=[[-10, 10], [-50, 50]]
Hx, xxedges, yxedges = np.histogram2d(X,Z,bins=nbinsx)
#Hx, xxedges, yxedges = np.histogram2d(X,PX2,bins=nbinsx)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax.set_xlabel('X [mm]',fontsize=20)
ax.set_ylabel('X\' [mrad] ',fontsize=20)
ax.grid(True)




HMAX=np.max(Hx)
# Mask zeros
xmid = 0.5*(xxedges[1:] + xxedges[:-1])
ymid = 0.5*(yxedges[1:] + yxedges[:-1])
# Calculate averages
# Creamos una malla de coordenadas para los puntos medios
xv, yv = np.meshgrid(xmid, ymid, indexing='ij')

# Calcula la media ponderada para x e y
mean_x = np.sum(Hmaskedx * xv) / np.sum(Hmaskedx)
mean_y = np.sum(Hmaskedx * yv) / np.sum(Hmaskedx)
print("Media en x: ",mean_x,np.mean(PX))
print("Media en y: ",mean_y,np.mean(X))
# Desviacin estndar ponderada para x e y
std_x = np.sqrt(np.abs(np.sum(Hmaskedx * (xv - mean_x)**2) / np.sum(Hmaskedx)))
std_y = np.sqrt(np.abs(np.sum(Hmaskedx * (yv - mean_y)**2) / np.sum(Hmaskedx)))
std_yx = np.sqrt(np.abs(np.sum(Hmaskedx * (yv - mean_y)*(xv - mean_x)) / np.sum(Hmaskedx)))

print("Desviacion estandar en x: ",std_x,np.std(PX))
print("Desviacion estandar en y: ",std_y,np.std(X),std_yx)
#print(xmid)
Emit=np.sqrt(std_x*std_y-std_yx*std_yx)
print("La emitancia es ",1*Emit," mm.mrad")

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)  

#fig, ax = plt.subplots()
#ax.hist(PZ, bins = 40)
fig, axre= plt.subplots(ncols=1, sharey=True)
#axre.hist(PY, bins = 200,alpha=0.5, color='orange',density=True,label='P$_Z$')
#axre.hist(PT, bins = 200,alpha=0.5, color='green',density=True,label='P$_Z$')
axre.scatter(PX,PY)
plt.show()
