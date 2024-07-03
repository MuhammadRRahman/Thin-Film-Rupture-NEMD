#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:08:52 2024

@author: muhammadrizwanurrahman
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
ppdir = './'
sys.path.append(ppdir)
import postproclib as ppl
from functions_thinfilms import colormap_sopfilm, setPlotParams
from scipy.signal import savgol_filter
from matplotlib.patches import Circle 
import scipy.ndimage 


def detect_hole (densitydata, dx):
    
     """
     From density data, deduce 2D thickness array, and retuns
     rho: 3D array (x, y, z) at t  
     """
     rho = densitydata 
     
     densitycut = 0.1
     holecutoff = 0.4
     # Compute the liquid region
     liquid = np.array(rho > densitycut, dtype=int)

     # Compute the interface regions
     interface = np.gradient(liquid, axis=0)

     # Initialize thickness matrix
     thickness = np.zeros((interface.shape[1], interface.shape[2]))
     
     numOfholes, hole_centers = np.nan, np.nan 
     # Compute the thickness for each point
     for i in range(interface.shape[1]):
         for j in range(interface.shape[2]):
             # Should be 2 values for top and bottom surface
             indx = np.where(interface[:,i,j] != 0)[0]
             if len(indx) != 0:
                 thickness[i,j] = dx * (indx[-1] - indx[0])
             else:
                 thickness[i,j] = 0.

     # Smooth the thickness
     smoothedthickness = thickness.copy() 
     

     # Binarize the thickness (with holes as 1 and liquid as 0)
     binarythickness = smoothedthickness.copy()
     binarythickness[binarythickness<=holecutoff] = 1        
     binarythickness[binarythickness>1] = 0 

     # Compute the label matrix and the number of holes
     s = [[0,1,0], [1,1,1], [0,1,0]]  # connectedness of the holes
     lw, numOfholes = scipy.ndimage.label(binarythickness, structure=s)

     # Ensure lw and binarythickness have the same shape
     if lw.shape != binarythickness.shape:
         raise ValueError("Mismatch in shapes of lw and binarythickness")
     
     # Calculate the centers of mass of the holes
     hole_centers = scipy.ndimage.center_of_mass(binarythickness, lw, range(1, numOfholes+1))

     return numOfholes, hole_centers, smoothedthickness
        



def get_density (fdir, nrec = 1000):
    
    """
    From simulation rho data, find 4D density arrays
    """
    
    fname = 'rho' # field to analyse
    PPObj = ppl.All_PostProc(fdir)   
    
    #Get plotting object
    plotObj = PPObj.plotlist[fname] 
    x, y, z = plotObj.grid 
    Y,Z = np.meshgrid(y,z) 
    dx = float(plotObj.header.binsize1)
    dy = float(plotObj.header.binsize2)
    dz = float(plotObj.header.binsize3)
     
    
    tplot = float(plotObj.header.tplot) 
    mass_avg_steps = float(plotObj.header.Nmass_ave) 
    deltat = float(plotObj.header.delta_t) 
    time_factor = tplot * deltat*mass_avg_steps 
    
     
    startrec = 1
    endrec =   plotObj.maxrec-1
    endrec = min(nrec,endrec)
    
    
    try:
        densityObj = PPObj.plotlist['rho']
        densityData = densityObj.read(startrec=startrec, endrec=endrec) # [x, y, z, t, comp]
    

        densityData = densityData[:, :, :, :, 0] # [x, y, z, t]
                
        return densityData , dx, dy, dz, time_factor 
    
    except FileNotFoundError :
       print("No files in directory !")
       
       return 
   
 

""" USER DEFINED VAARIABLES BASED (CASE SPECIFIC) """

cmap = colormap_sopfilm()

# directory
fdir0 = './32-192-192-no-poke-en02/'
fdir = fdir0 + 'results/'

# variables 
nrec = 200  
smoothfactor = 1

""" INITIALIZING OUTPUT VARIABLES """
numOfholes = 0 
tn = np.nan 
hole_centers = np.array([np.nan,np.nan])


""" MAIN TASK STARTS HERE """

# get density array from results 
density_data, dx, dy, dz, time_factor = get_density (fdir,nrec) 
dlen = density_data.shape[-1]
nrec = min(dlen,nrec)

numOfholes = 0
for rec in range(0,nrec):
    
    
    """ Detect hole on the film """
    density_at_rec = density_data[:,:,:,rec]
    numOfholes, hole_centers, smoothedthickness = detect_hole (density_at_rec, dx) 

    
    if (numOfholes >0) and np.isnan(tn):
        tn = rec  # record time of nucleation 
        print(f'\n tn is: {tn}')
        break 
    
        
    """ Visualize the film thickness """ 
    plt.imshow(smoothedthickness,cmap=cmap, vmin=0,vmax=20,origin='lower')
    
    if numOfholes >0:
    
        hole_location = np.array([ hole_centers[0][1], hole_centers[0][0] ]) 
        circle = Circle(hole_location, radius=5, color='r', fill=False, linewidth=3)
        # Add the circle to the plot
        plt.gca().add_patch(circle)
    
        setPlotParams() 
        plt.title(str(rec)) 
        plt.show() 


hole_location = np.array([ hole_centers[0][1], hole_centers[0][0] ]) 


    
         
        
        

