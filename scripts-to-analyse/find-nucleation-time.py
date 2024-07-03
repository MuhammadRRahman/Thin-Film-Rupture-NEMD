#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:28:00 2023
@author: muhammadrizwanurrahman

Inspect all the spontaneous cases, find the initial mean thickness, h0, nucleation time, tn1
and save in pandas' dataframe format in a .csv file. 
saved file contains: 
    (1) 'initial_thickness_in_MD', (2) 'time_of_nucleation_in_MD'
    (3) 'initial_step_number' , (3) 'time_factor' (4) 'file_directory'
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sys 
import matplotlib.pyplot as plt
ppdir = '../../Thin-Films-Functionfiles-2023/postproclib/'
sys.path.append(ppdir)
import postproclib as ppl
import pandas as pd


def get_density(PPObj, startrec, endrec, boundary = None, tavg=0):
    
    """
    # returns: densityData # [x, y, z, t] 
    """    
    try:
        densityObj = PPObj.plotlist['rho']
        densityData = densityObj.read(startrec=startrec, endrec=endrec) # [x, y, z, t, comp]
    
        if boundary is not None:
            ymin, ymax = boundary[0], boundary[1]
            zmin, zmax = boundary[2], boundary[3]
        
            densityData = densityData[:, ymin:ymax, zmin:zmax, :, 0] # [x, y, z, t]
        else:
            densityData = densityData[:, :, :, :, 0]
            
        if tavg > 0:
            # convolve with a spatial window
            window = np.ones((tavg*2+1,))/ (tavg*2+1) # flat window
            densityData = np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=-1, arr=densityData)
        
        return densityData  
    
    except FileNotFoundError :
       print("No files in directory !")
       return 
    
from scipy.ndimage import measurements
def get_thickness(dendata, dx):
    
    """
    rho: 3D array (x, y, z) at time_step t
    dx: bin size along film thickness, float
    
    returns: 
        numOfholes, smoothedthickness in units of cells instead of MD
    """
    rho = dendata.copy() 
    
    densitycut = 0.1
    holecutoff = 0.4
    # Compute the liquid region
    liquid = np.array(rho > densitycut, dtype=int)
    # Compute the interface regions
    interface = np.gradient(liquid, axis=0)    
    # Initialize thickness matrix
    thickness = np.zeros((interface.shape[1], interface.shape[2]))
    
    numOfholes = np.nan 

    # Compute the thickness for each point
    for i in range(interface.shape[1]):
        for j in range(interface.shape[2]):
            # Should be 2 values for top and bottom surface
            indx = np.where(interface[:,i,j] != 0)[0]
            if len(indx) != 0:
                thickness[i,j] = dx*(indx[-1] - indx[0])
            else:
                thickness[i,j] = 0.
    
    # Smooth the thickness
    smoothedthickness = thickness.copy()   # this is in MD units    
    # Binarize the thickness (with holes as 1 and liquid as 0)
    binarythickness = smoothedthickness.copy()
    binarythickness[binarythickness<=holecutoff] = 1        
    binarythickness[binarythickness>1] = 0 

    # Compute the label matrix and the number of holes
    s = [[0,1,0], [1,1,1], [0,1,0]]  # connectedness of the holes
    lw, numOfholes = measurements.label(binarythickness, structure=s)

    # Ensure lw and binarythickness have the same shape
    if lw.shape != binarythickness.shape:
        raise ValueError("Mismatch in shapes of lw and binarythickness")
    
    return numOfholes, smoothedthickness 



"""
## ************************************************************************
                         MAIN CODE STARTS HERE
## ************************************************************************
"""



fdir0 = '/Volumes/MRR_03/paper-03-spontaneous/no-poke-confirming-film-stability-A/' 
fdirs = [
         '20-192-192-no-poke-en01','20-192-192-no-poke-en02','20-192-192-no-poke-en03',
         '22-192-192-no-poke-en01','22-192-192-no-poke-en01','22-192-192-no-poke-en01',
         '24-192-192-no-poke-en01','24-192-192-no-poke-en02','24-192-192-no-poke-en03',
         '26-192-192-no-poke-en01','26-192-192-no-poke-en02','26-192-192-no-poke-en03',
         '28-192-192-no-poke-en01','28-192-192-no-poke-en02','28-192-192-no-poke-en03',
         '30-192-192-no-poke-en01','30-192-192-no-poke-en02','30-192-192-no-poke-en03',
         '31-192-192-no-poke-en01','31-192-192-no-poke-en02','31-192-192-no-poke-en03',
         '32-192-192-no-poke-en01','32-192-192-no-poke-en02','32-192-192-no-poke-en03',
         '33-192-192-no-poke-en01','33-192-192-no-poke-en02','33-192-192-no-poke-en03',
         '34-192-192-no-poke',
         '36-192-192-no-poke',
         '38-192-192-no-poke'
         ] 

datadir = '/Users/muhammadrizwanurrahman/MRR_Research/gitREPO-data-and-images/spontaneous-nucleation/all-that-is-good-data/thickness-and-nucleation-time/'
filename = datadir + 'thickness-and-nucleation-time-svf.csv'

initial_thickness = [] 
time_of_nucleation = [] 
initial_step = []
time_mutiplying_factor = []
file_directory = [] 

for f in fdirs:
    
    try:
        fdir = fdir0 + f + '/results/'
        
        PPObj = ppl.All_PostProc(fdir)
        plotObj = PPObj.plotlist['rho']
        dx = float(plotObj.header.binsize1)
        rec0 = float(plotObj.header.initialstep)
        tplot = float(plotObj.header.tplot)
        mass_avg_steps = float(plotObj.header.Nmass_ave)
        deltat = float(plotObj.header.delta_t)    
        time_factor = tplot*mass_avg_steps*deltat
        
        boundary = None  
        startrec, endrec = 0,  plotObj.maxrec-1
        tavg = 1 # time window over which all data will be temporaly averaged (moving average)
    
        # get_density(PPObj, startrec, endrec, boundary = None, tavg=0)
        rho = get_density(PPObj, startrec, endrec, boundary, tavg=tavg)
        
        # initialize the variables
        num_of_holes = np.nan 
        h0 = np.nan 
        tn1 = np.nan 
        
        for rec in range(startrec,endrec):
            
            num, film_thickness = get_thickness(rho[:,:,:,rec], dx) 
            if rec ==startrec:
                h0 = film_thickness.mean()
            
            if num >=1:
                tn1 = rec * time_factor  
                
                break 
            
        initial_thickness.append(h0)
        time_of_nucleation.append(tn1) 
        initial_step.append(rec0)
        time_mutiplying_factor.append(time_factor) 
        file_directory.append(fdir)
        
        
    except FileNotFoundError:
        print(f"No files in directory '{filename}'") 
        continue 
 
data2save = {'initial_thickness_in_MD': initial_thickness, 
             'time_of_nucleation_in_MD': time_of_nucleation,
             'initial_step_number': initial_step, 
             'time_factor': time_mutiplying_factor,
             'file_directory' : file_directory
             }  

df = pd.DataFrame(data2save)

df.to_csv(filename, index=False )    
print(f"DataFrame saved as CSV file: '{filename}'")













