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
ppdir = './postproclib/'
sys.path.append(ppdir)
import postproclib as ppl
from scipy.signal import savgol_filter


def get_thickness (fdir):
    
    fname = 'rho' # field to analyse
    PPObj = ppl.All_PostProc(fdir)   
    
    #Get plotting object
    plotObj = PPObj.plotlist[fname]
    x, y, z = plotObj.grid
    Y,Z = np.meshgrid(y,z)
    ha = 1
    dx = float(plotObj.header.binsize1)
    dy = float(plotObj.header.binsize2) 
    dz = float(plotObj.header.binsize3) 
    
    tplot = float(plotObj.header.tplot) 
    mass_avg_steps = float(plotObj.header.Nmass_ave) 
    deltat = float(plotObj.header.delta_t) 
    time_factor = tplot * deltat*mass_avg_steps 
    
    print(time_factor)
    
    rho = np.mean(plotObj.read(0, 17),3) # testing on some initial time-steps only
    densitycut = 0.2 
    liquid = np.array(rho[:,:,:,0] > densitycut,dtype=int)
    interface = np.gradient(liquid[:,:,:],axis=0)
    
    # Main task starts here   
    thickness = np.zeros([interface.shape[1], interface.shape[2]])
    top_surface = np.zeros([interface.shape[1], interface.shape[2]])
    bottom_surface = np.zeros([interface.shape[1], interface.shape[2]])
    startrec = 0 # assigning to zero
    ha = 1
    
    endrec =  plotObj.maxrec-1
    h0 = np.nan 
    ha = 0
    startrec = 0+ha
    
    records = np.arange(0,min(endrec,1000),1)
    
     
    smoothfactor = 1 # used 4 for smooth images
    num_steps = len(records)
    H = np.zeros((thickness.shape[0]*smoothfactor,thickness.shape[1]*smoothfactor,num_steps))
    topZ =  np.zeros((thickness.shape[0]*smoothfactor,thickness.shape[1]*smoothfactor,num_steps))
    botZ =  np.zeros((thickness.shape[0]*smoothfactor,thickness.shape[1]*smoothfactor,num_steps))
    time_in_lj = []
    
    for rec in np.arange(startrec,len(records),1):
        
        
        rec = int(rec)
        rho = np.mean(plotObj.read(rec-ha, rec+ha),3)
        liquid = np.array(rho[:,:,:,0] > densitycut,dtype=int)
        interface = np.gradient(liquid[:,:,:],axis=0)
        
        for i in range(interface.shape[1]):
            for j in range(interface.shape[2]):
                #Should be 2 values for top and bottom surface
                indx = np.where(interface[:,i,j] != 0)[0]
                if len(indx) != 0:
                    thickness[i,j] = dx * (indx[-1] - indx[0])
                    top_surface[i,j] = dx * (indx[-1]) 
                    bottom_surface[i,j] = dx * (indx[0])
                else:
                    thickness[i,j] = 0.  
                    top_surface[i,j] = 0. 
                    bottom_surface[i,j] = 0.
               
        smoothedthickness = zoom(thickness, smoothfactor)
        
        H [:,:,rec] = smoothedthickness
        topZ[:,:,rec] = top_surface 
        botZ[:,:,rec] = bottom_surface
        time_in_lj.append(rec*time_factor)
        
        
        if rec == startrec :
            #film_thickness_initial, h0
            h0 = round(smoothedthickness.mean(),2) # already converted to MD units by multiplying by dx
            print(f'\n h0 is: {h0}')
            
    time_in_lj = np.array(time_in_lj)        
    
    return H, h0, time_in_lj, topZ, botZ, dy, dz, time_factor


def detect_hole (dendata, dx, stop_at_first_hole=1):
    
    """
    rho: 3D array (x, y, z) at time_step t
    dx: bin size along film thickness, float
    
    returns: 
        numOfholes, smoothedthickness in units of cells instead of MD,
        hole_coordinates: list of coordinates for each hole
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
    hole_coordinates = []

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
    
    # Find coordinates of each hole
    for label in range(1, numOfholes + 1):
        coordinates = np.argwhere(lw == label)
        hole_coordinates.append(coordinates)
        
     
    if (numOfholes) :
        print('\n holes num',numOfholes)
        
        if (numOfholes==1):
            print ('\n holes at:',hole_coordinates)
               
        
    return numOfholes, hole_coordinates



def calculate_U_2D (H, dx, dy, h0=None):
    """
    Calculate the U(h) for each time step in a 2D film.

    Parameters:
    - H: 3D numpy array of film thickness (x, y, t).
    - dx: The grid spacing in x-, and y-direction.

    Returns:
    - U_over_time: 1D numpy array of U values over time.
    """
    U_over_time = np.zeros(H.shape[2])
    
    
    # Iterate over each time step
    for t in range(H.shape[2]):
        # Extract the film at time t
        h_t = H[:, :, t]

        # Calculate the partial derivatives
        dhdx = np.gradient(h_t, dx, axis=1)
        dhdy = np.gradient(h_t, dy, axis=0)


        A = 1e-19 # J 
        epsilon = 1.65 * 1e-21 # energy scale
        A_reduced = A/epsilon # in reduced units since h is in reduced units
        
        if h0 is not None:
            U_dim = (0.5/2) * (dhdx**2 + dhdy**2) - (A_reduced / (6*np.pi* h_t**4)) * (h_t - h0)**2 
        else:    
            U_dim = (0.5/2) * (dhdx**2 + dhdy**2) - (A_reduced / (6*np.pi* h_t**2))

        # Integrate U over the domain to get the total energy at time t
        U_over_time[t] = np.sum(U_dim) * dx * dy

    return U_over_time




def get_U_over_time(H, hole_location, safety_margin, dx, dy, h0):
    """
    Calculate U(h) around the hole location for each time step.

    Parameters:
    - H: 3D numpy array of film thickness (x, y, t).
    - hole_location: Tuple of (x, y) indicating the hole location.
    - safety_margin: Number of indices around the hole location to consider.
    - dx, dy: The grid spacing in the x, and y-direction.

    Returns:
    - U_values: 1D numpy array of U values over time.
    """
    x, y = hole_location
    # Define the slice for the region of interest
    slice_x = slice(max(0, x - safety_margin), min(H.shape[0], x + safety_margin + 1))
    slice_y = slice(max(0, y - safety_margin), min(H.shape[1], y + safety_margin + 1))

    # Extract the region of interest
    H_region = H[slice_x, slice_y, :]

    # Calculate U for the region over time
    U_values = calculate_U_2D(H_region, dx, dy, h0)
    
    return U_values




def get_interpolated (original_array, number_of_points):
    from scipy.interpolate import interp1d

    original_indices = np.linspace(0, 1, num=len(original_array), endpoint=True)

    # New indices for the interpolated array
    target_indices = np.linspace(0, 1, num=number_of_points, endpoint=True)
    
    # Create the interpolation function
    interp_function = interp1d(original_indices, original_array, kind='linear')
    
    # Interpolate the array
    interpolated_array = interp_function(target_indices)
    
    return interpolated_array


# DATA DIRECTORIES
# REPLACE THE PATHS WITH THE PATHS TO YOUR DIRECTORIES CONTAINING DATA DENSITY DATA
# FROM FLOWMOL SIMULATIONS

fdir0 = './data-dir/'
fdirs = ['dir-1/', 'ddir-2/', 'dir-3/' , 'dir-4/' ,'dir-5/','dir-6/','dir-7/','dir-8/']

# REPLACE THE HOLE LOCATIONS AS OF YOUR CASES 
# -------------------------------------------------------------------------------------------------
# The hole locations are found by separately calling the detect_hole function from above.
# The cases given here as for demo, all the cases that are re-started within the memory window 
# ruptured at (34,19), the films restarted from an earlier point in time did not rupture at (34,19),
# hence the location for these cases are replaced by (np.nan, np.nan)
hole_locations = [ (35,20), (np.nan, np.nan), (np.nan, np.nan),  
                   (34,19), (34,19), (34,18), (34,19), (35,19), 
                 ]
# np.nan means these points did not rupture at the parent locations

# simulation restart time of each case: iteration number x factor-to-convert-to-time 
t0s = np.array([80312, 85791, 89964, 94376, 94376,94376,94376,94376]) * 0.005 

fig,ax = plt.subplots(figsize=(10,6))
hole_location = (np.nan, np.nan)

case = 0
for fdir1 in fdirs:
        
    fdir = fdir0 + fdir1 + 'results/'
        
    t0 = t0s[case]   
    nrec = nrecs[case]
    hole_location = hole_locations[case]
    color = colors [case]
    alpha = alphas[case]
    
    case+=1 
  
    # if this film nucleates at other location, still examine the site under inspection 
    sym = 'o'
    if np.isnan(hole_location[0]):
        hole_location = (34,19)
        sym = 'x'
    
    H, h0, time_in_lj, topZ, botZ, dy, dz, time_factor = get_thickness (fdir)

    safety_margin=3 # checking the 3x3 neighbourhood of the rupture site
    
    U_values = get_U_over_time(H[:,:,0:nrec], hole_location, safety_margin, dy, dz, h0)
    
    time =  t0 + np.arange(0,nrec) * time_factor
    
    time_finer = get_interpolated (time  , number_of_points=300)
    U_finer = get_interpolated (U_values , number_of_points=300)
    U_finer_sm = savgol_filter(U_finer,20,3)    
           
    ax.plot(t0,U_finer_sm[0],'o',ms=18, mfc=color,mec='k',mew=3)
    ax.plot(time_finer[-1],U_finer_sm[-1],sym,ms=18, mfc='w',mec=color,mew=3)


figname = './U-JS.jpg'
plt.savefig(figname,dpi=600)
plt.show() 

 





