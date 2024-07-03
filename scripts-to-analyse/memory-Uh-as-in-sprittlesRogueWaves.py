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
#ppdir = '/home/muhammad/Research/GitMRR/Thin-Films-Functionfiles-2023/postproclib/'
ppdir = '../Thin-Films-functionfiles-2023/postproclib/'
sys.path.append(ppdir)
import postproclib as ppl
from functions_thinfilms import colormap_sopfilm, setPlotParams
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



def calculate_Hessian_Energy(H, dx, dy):
    """
    Calculate the Hessian of Energy for each time step in a 2D film.

    Parameters:
    - H: 3D numpy array of film thickness (x, y, t).
    - dx: The grid spacing in x-, and y-direction.

    Returns:
    - U_over_time: 1D numpy array of U values over time.
    """
    
    HofU_over_time = np.zeros(H.shape[2])

    # Iterate over each time step
    for t in range(H.shape[2]):
        # Extract the film at time t
        h_t = H[:, :, t]

        # Calculate the partial derivatives
        dhdxx = np.gradient(np.gradient(h_t, dx, axis=1), dx, axis=1)
        dhdyy = np.gradient(np.gradient(h_t, dy, axis=0), dy, axis=0)

        # Calculate U(h) using the equation
        HofU = 0.5 * (dhdxx**2 + dhdyy**2) - ((2 * np.pi**2) / (3 * h_t**3))

        # Integrate U over the domain to get the total energy at time t
        HofU_over_time[t] = np.sum(HofU) * dx * dy

    return HofU_over_time 


def calculate_U_2D (H, dx, dy, h0=None):
    """
    Corrected based on J.S.'s comments'
    Calculate the U(h) for each time step in a 2D film.

    Parameters:
    - H: 3D numpy array of film thickness (x, y, t).
    - dx: The grid spacing in x-, and y-direction.

    Returns:
    - U_over_time: 1D numpy array of U values over time.
    """
    # Preallocate the U array for performance
    U_over_time = np.zeros(H.shape[2])
    #C = 1e-19/(12*3.1419)
    
    
    # Iterate over each time step
    for t in range(H.shape[2]):
        # Extract the film at time t
        h_t = H[:, :, t]

        # Calculate the partial derivatives
        dhdx = np.gradient(h_t, dx, axis=1)
        dhdy = np.gradient(h_t, dy, axis=0)

        # Calculate U(h) using the equation
        #U = 0.5 * (dhdx**2 + dhdy**2) - ((2 * np.pi**2) / (3 * h_t**2))
        #U = 0.5 * (dhdx**2 + dhdy**2) - (1 / h_t**2)
        
        C = (2 * np.pi**2) / 3
        A = 1e-19 # J 
        epsilon = 1.65 * 1e-21 # energy scale
        A_reduced = A/epsilon # in reduced units since h is in reduced units
        #gamma = 0.5
        #chi = A/ (8*np.pi**3/gamma)
        #chi = 1 
        #U = 0.5 * (dhdx**2 + dhdy**2) - (C / h_t**2)
        #U_dim = (0.5/chi) * (dhdx**2 + dhdy**2) - (C*chi / h_t**2)
        #U_dim = (0.5/2) * (dhdx**2 + dhdy**2) - (A_reduced / (6*np.pi* h_t**2))
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

############################################
def calculate_tau(U_values, epsilon=0.34, C=1):
    """
    Calculate the expected rupture time tau based on Kramer's law, given the U values over time.

    Parameters:
    - U_values: 1D numpy array of U values over time.
    - Delta: Noise strength parameter.
    - C: Prefactor in Kramer's law (default is 1, actual value needs to be calculated based on additional procedures).

    Returns:
    - tau: The expected rupture time.
    """
    # Assuming U(h_0) is the initial energy and U(h_s) is the minimum energy observed
    U_h0 = U_values[0]
    U_hs = -10.888382002686049
    
    tau = C * np.exp(-1/epsilon * (U_hs - U_h0))
    return tau

def calculate_and_get_tau(H, hole_location, safety_margin, dx, dy, epsilon=0.34, C=1, h0=None):
    """
    Full workflow to calculate U(h) over time and then use it to calculate tau.

    Parameters:
    - H: 3D numpy array of film thickness (x, y, t).
    - hole_location: Tuple of (x, y) indicating the hole location.
    - safety_margin: Number of indices around the hole location to consider.
    - dx: The grid spacing in the x-direction.
    - dy: The grid spacing in the y-direction.
    - Delta: Noise strength parameter.
    - C: Prefactor in Kramer's law.

    Returns:
    - tau: The expected rupture time.
    """
    # Calculate U over time
    U_values = get_U_over_time(H, hole_location, safety_margin, dx, dy, h0)
    
    # Calculate tau
    tau = calculate_tau(U_values, epsilon, C)
    
    return tau
############################################


fdir0 = '/Volumes/MRR-04/Project-0203-Film-Rupture/paper-03-spontaneous/new_natural_savefinals/spont-36-192-savefinals/run_interim_80312_en04_savefinals_child_mother/'
#fdirs = ['results_child_mother/', 'child_085791_en03/', 'child_089964_en02/','child_094376_en04/','child_100878_en02/','child_104094_en03/']

fdirs = ['results_child_mother/', 
         
         'child_085791_en03/',#'child_085791_en02/','child_085791_en01/',
         
         'child_089964_en02/' ,#'child_089964_en01/', 'child_089964_energy_01/',
         
         'child_094376_en05/' ,#'child_094376_en04/','child_094376_en03/', 'child_094376_en02/', 'child_094376_en01/',
         
         'child_100878_en03/' ,# 'child_100878_en02/', 'child_100878_en01/', 
         
         'child_104094_en03/' ,# 'child_104094_en02/', 'child_104094_en01/'
         ]


hole_locations = [ (35,20),
                   (np.nan, np.nan), #(np.nan, np.nan), (np.nan, np.nan), #  (16,29), (29,28), (17,30),
                   (np.nan, np.nan), #(np.nan, np.nan), (np.nan, np.nan), #  (9,12),  (9,12),  (41,43),
                   (34,19), #(34,19), (34,18), (34,19), (35,19), 
                   (35,19), #(34,19), (35,19),
                   (35,20), #(35,20), (34,19) 
    ]
# np.nan means these points did not rupture at the parent locations


nrecs = np.array([
        34,
        36, #31, 29,
        40, #35, 40, 
        7, #7, 7, 7 ,7,
        8, #8, 8, 
        9, #9, 9,
        ]) 

t0s = np.array([
      80312,
      85791,#85791,85791,
      89964,#89964,89964,
      94376,#94376,94376,94376,94376,
      100878,#100878,100878,
      104094,#104094,104094,     
      ]) * 0.005 



fig,ax = plt.subplots(figsize=(10,6))
setPlotParams()
colors = [ 
             'black',
             'silver',#'silver','silver',
             'gray',#'gray','gray',
             'red',#'red','red','red','red',
             'firebrick',#'firebrick','firebrick',
             'navy',#'navy','navy'
          ]

alphas = [1,
          1, #0.6, 0.2,
          1, #0.6, 0.2,
          1, #0.7, 0.5, 0.3, 0.2,
          1, #0.6, 0.2,
          1, #0.6, 0.2,
          ]


taus = []


hole_location = (np.nan, np.nan)
case = 0 
 

for fdir1 in fdirs:
    
    if fdir1 == fdirs[0]:
        fdir = fdir0 + fdir1 
        
    else: 
            
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

    safety_margin=3
    
    U_values = get_U_over_time(H[:,:,0:nrec], hole_location, safety_margin, dy, dz, h0)
    
    time =  t0 + np.arange(0,nrec) * time_factor
    
    U_values [U_values < -100] = -105 # to void large infinite negative number 
    #time [U_values < -100] = np.nan 
    
    time_finer = get_interpolated (time  , number_of_points=300)
    U_finer = get_interpolated (U_values , number_of_points=300)
    #U_finer += np.random.uniform(-0.5, 0.5, U_finer.shape)
    U_finer_sm = savgol_filter(U_finer,20,3)
    

    
    if fdir1 == fdirs[3]:
        print(f'U_hs is: {U_values[0]}')
        U_hs = U_values [0]
    
    
    #####
    
    tau = calculate_and_get_tau(H, hole_location, safety_margin, dy, dz, epsilon=.34, C=1, h0=h0)
    taus.append(tau)
    print(f'\n tau is: {tau}')
    #####
    
    
    #fig,ax = plt.subplots(figsize=(10,6))
    dashedTh = -60
    ax.plot(time_finer[U_finer_sm > dashedTh] ,U_finer_sm[U_finer_sm >  dashedTh] ,'-', linewidth=5, alpha=alpha, color=color)
    ax.plot(time_finer[U_finer_sm < dashedTh] ,U_finer_sm[U_finer_sm <  dashedTh] ,'--', linewidth=5, alpha=alpha, color=color)
    
    ax.plot(t0,U_finer_sm[0],'o',ms=18, mfc=color,mec='k',mew=3)
    ax.plot(time_finer[-1],U_finer_sm[-1],sym,ms=18, mfc='w',mec=color,mew=3)
        

    #plt.xlim(initial_step,initial_step+)
    #ax.set_ylim(-50,50)
    ax.set_ylim(-50,50)
    #ax.set_yticks([-40,-20,0,20,40,60,80])
    ax.set_xlim(400,600)
    
    #ax.set_yscale('log')
    
    #kalpha +=1

figname = '/Volumes/MRR-04/Project-0203-Film-Rupture/paper-03-spontaneous/new_natural_savefinals/' + 'U-corrected-C0p01.jpg'
#plt.savefig(figname,dpi=600)
plt.show() 

 





