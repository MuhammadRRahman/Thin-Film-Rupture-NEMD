#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:19:22 2022
@author: MuhammadRRahman, edwardsmith999
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.path import Path

ppdir = './postproclib/' # path to postproclib
sys.path.append(ppdir)
import postproclib as ppl

# Font size and type
fsize = 20
plt.rc('font', size=fsize)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.size'] = 40
plt.rcParams['axes.labelsize'] = 38
plt.rcParams['font.family'] = 'Times New Roman' #'sans-serif'
plt.rcParams['axes.linewidth'] = 3 #set the value globally
plt.rcParams['lines.linewidth'] = 3 #set the value globally
plt.tick_params(axis='both', which='major', pad=20, length=10, width=2)

## ................................................................................ ##
from Func_surface_functions import radial_profile, order, transform_velocity
## ................................................................................ ##

normal =0
component=1
startrec= 0
ha =  2
skip = 50
rhocutoff = 0.2
cmap = plt.cm.RdYlBu_r

#Get Post Proc Object
fdir = './results/' # path to flowmol data files
PPObj = ppl.All_PostProc(fdir)
print(PPObj)

#Get plotting object
rhoObj = PPObj.plotlist['rho']
Uobj = PPObj.plotlist['vbins']
vmn = None #-0.2
vmx = None # 0.05
Y2, Z2 = np.meshgrid(Uobj.grid[1],
                   Uobj.grid[2],indexing='ij')
endrec= Uobj.maxrec-ha

x = Uobj.grid[0]
y = Uobj.grid[1]
z = Uobj.grid[2]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

tplot = float(Uobj.header.tplot)
mass_avg_steps = float(Uobj.header.Nmass_ave)
deltat = float(Uobj.header.delta_t)

#Get Udata shape
Udata = Uobj.read(startrec=startrec, endrec=startrec)
mid = [int(Udata.shape[i]/2.)-1 for i in range(3)]
cnt = 0

length_with_velocity = []
startrec =0
for rec in range(startrec+ha, endrec, skip):
        
        # First plot all the slices, then the radial averaged rotated velocity
        #fig, ax = plt.subplots(figsize=(10,10)) ## to plot contours of velocity profiles 
        #fig,ax = plt.subplots(figsize=(10,10))
        #fig,ax = plt.subplots(1,2,figsize=(8,8))
        rave = []; Vave = []; Vave2 = [];
        
        ### 
        rho = rhoObj.read(startrec=rec-ha, endrec=rec+ha)
        rho = np.mean(rho[:,:,:,:,0],3)
        #Average around in a radial sweep
        for i in range(Udata.shape[0]):
            rave.append(radial_profile(rho[i,:,:],[mid[1],mid[2]]))
        rave = np.array(rave)
    
        maxr = np.sqrt(y.max()**2+z.max()**2)
        rad = np.linspace(0.,maxr,rave.shape[1])
        dR = np.diff(rad)[0]
        Xr, R = np.meshgrid(x, rad, indexing="ij")
        
        #Get outer surface coordinates
        mask = rave > rhocutoff
        difvalue=(np.abs(np.diff(np.array(mask,dtype=int),axis=0))[:,:-1]+ 
                  np.abs(np.diff(np.array(mask,dtype=int),axis=1))[:-1,:]) > 0.
        coordinates=np.where(difvalue==True)
        #Add first value to end so it loops around
        xp=Xr[coordinates[0]+1, coordinates[1]+1]    # y coordinates for fitting 
        rp= R[coordinates[0]+1, coordinates[1]+1]    # z coordinates for fitting 
    
        r = order(xp, rp); Np = r.shape[0]
        #ax.plot(r[:,0], r[:,1], 'r-', linewidth = 2)
        #ax[1].plot(r[:,0], r[:,1], 'k--', linewidth = 1)
        
        
        # velocity : average around in a radial sweep
        Udata = Uobj.read(startrec=rec-ha, endrec=rec+ha)
        Udata = np.mean(Udata[:,:,:,:,:],3)
        
        # transform the velocity field in the radio-direction
        UR, UT = transform_velocity(Udata[:,:,:,1], Udata[:,:,:,2], Y2, Z2)
        dt = Udata[:,:,:,0]
        Vave_x = []
        for j in range(Udata.shape[0]):
            Vave_x.append(radial_profile(dt[j,:,:],[mid[1],mid[2]]))
        Uradial_x = np.array(Vave_x)

        dvr = UR
        Vave_R = []
        for j in range(Udata.shape[0]):
            Vave_R.append(radial_profile(dvr[j,:,:],[mid[1],mid[2]]))
        Uradial_R = np.array(Vave_R)

 
        
        ## *************************************************************
        ##                      plotting temperature
        ## ************************************************************* 
        vmn = 0
        vmx = 1.0
        fig,ax = plt.subplots(figsize=(8,8))
        #f,ax2 = plt.subplots(figsize=(10,10))
        TObj = PPObj.plotlist['T_peculiar']
        T = TObj.read(startrec=rec-ha, endrec=rec+ha)
        T = np.mean(T[:,:,:,:,0],3)
        Tave = []
        #Average around in a radial sweep
        #figT,axT = plt.subplots(figsize=(8,8))
        
        for k in range(Udata.shape[0]):
            Tave.append(radial_profile(T[k,:,:],[mid[1],mid[2]]))
        Tave = np.array(Tave)
        ## NORMALIZED TEMPERATURE FIELD
        # cT = ax.pcolormesh(Xr, R, Tave/np.max(Tave), cmap=plt.cm.RdYlBu_r,shading='gouraud',vmin=vmn,vmax=vmx)
        ## Non-NORMALIZED TEMPERATURE FIELD
        # cT = ax.pcolormesh(Xr, R, Tave, cmap=plt.cm.RdYlBu_r,shading='gouraud',vmin=vmn,vmax=vmx)
        
        ## TEMPERATURE FIELD IN UNITS OF KELVIN
        epsilon = 1.65e-21
        kB = 1.380649e-23
        Tave_K = Tave * epsilon / kB # in Kelvin
        Tave_K [Tave_K<50] = 0
        Tave_K [Tave_K>100] = 100
     
        # change Tmin and Tmax according to data range
        print('Temperature clipped by Tmin = 80 and Tmax = 95') 
                      
        cT = ax.pcolormesh(Xr, R, Tave_K, cmap='plasma',vmin=85,vmax=95)
        plt.pcolormesh(Xr, R, Tave_K, cmap=plt.cm.RdYlBu_r,shading='gouraud',vmin=85,vmax=90)
        ax.set_xlim([-20,20])
        ax.set_ylim([5,250])
        ax.set_aspect(1)
        plt.colorbar()
        
        
        #ax.plot(r[:,0]-0.1, r[:,1], 'r-', linewidth = 2)
        #ax2.set_xlim([-20,20])
        #ax2.set_ylim([60, 180])
        #ax2.set_aspect(1)
        #plt.colorbar(cT)
        
        ## *************************************************************
        ##              contour and quiver plot of radial velocity
        ## *************************************************************
        """
        vmn = -0.2
        vmx = 1.2
        #fig,ax = plt.subplots(figsize=(8,8))
        #cmap = plt.cm.RdYlBu_r
        #cm=ax.pcolormesh(Xr, R, Vave_R/np.max(Vave_R), cmap=cmap, vmin=vmn, vmax=vmx,shading='gouraud') 
        #cm=plt.imshow(Vave_R/np.max(Vave_R),interpolation = 'bicubic',vmin=-0.5,vmax=1.5)
        
        
        ## quiver plot 
        
        U = Vave_x
        V = Vave_R
        w = 7/1000
        Q = ax.quiver(Xr, R, U, V,width=w)

        ax.set_xlim([-20,20])
        ax.set_ylim([-5, 300])
        
        timeis = rec * tplot * deltat *  mass_avg_steps
        title = 'rec: ' + str(rec) + 'time: ' + str(timeis)
        ax.set_title(title)
        h0 = r[:,0][-1]-r[:,0][0]
        plt.colorbar(cT)
        plt.show()
        """
        
       
