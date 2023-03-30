#!/usr/bin/env python3
# author: MuhammadRRahman, edwardsmith999
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile        


ppdir = './postproclib/'
sys.path.append(ppdir)
import postproclib as ppl


## ------------------------------------------------------------ ##
##                      USER INPUT                              ##
## ------------------------------------------------------------ ##

# field to analyse
fname = 'rho' 
fdir = './results/'
#Frequrncy of plot
skip = 2
#How many samples to average over in time
#useful if time resolution was too high
ave = 1 
if ave >1:
    startrec = int(ave/2)
else:
    startrec = int(ave)    

ha = int(ave/2.)


## ............................................................ ##
##  Calculate and plot radial average of density field          ##
## ............................................................ ##

outrec = 0 
#Get Data
PPObj = ppl.All_PostProc(fdir)
print(PPObj)

#Get plotting object
plotObj = PPObj.plotlist[fname]
x, y, z = plotObj.grid
Y,Z = np.meshgrid(y,z)
dx = float(plotObj.header.binsize1)
rec = plotObj.maxrec - ha - 1

fnameS = np.mean(plotObj.read(startrec=rec-ha, endrec=rec+ha),3)
mid = [int(fnameS.shape[i]/2.)-1 for i in range(3)] # ad
endrec = plotObj.maxrec-ha-1
dx = float(plotObj.header.binsize1)
dy = float(plotObj.header.binsize2)
figaspect =  .2 


for rec in np.arange(startrec,endrec,skip):
    
    fig, ax = plt.subplots(figsize=(8,8))
    rave = []
    fname = np.mean(plotObj.read(startrec=rec-ha, endrec=rec+ha),3)
        
    for i in range(fname.shape[0]):
        rave.append(radial_profile(np.mean(fname[i,:,:,:],2),[mid[1],mid[2]]))

    rave = np.array(rave)
        
    cm = ax.pcolormesh(rave, cmap=plt.cm.RdYlBu_r,vmin=0, vmax=1)
    ax.set_aspect(figaspect)
    plt.show()
    
    
    
