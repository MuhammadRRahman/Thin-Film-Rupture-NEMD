#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:46:50 2022
@author: Muhammad R Rahman, Edward R. Smith
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.path import Path

ppdir = './postproclib/'
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
from P02_surface_functions import radial_profile, order
from P02_surface_functions import transform_pressure_tensor 
## ................................................................................ ##


normal =0
component=1
startrec= 0
ha =  2
skip = 50
rhocutoff = 0.4
cmap = plt.cm.RdYlBu_r

fdir = 'film-rupture-h1/'
PPObj = ppl.All_PostProc(fdir)
print(PPObj)

#Get plotting object
rhoObj = PPObj.plotlist['rho']
plotObj = PPObj.plotlist['pVA']
vmn = None
vmx = None
Y2, Z2 = np.meshgrid(plotObj.grid[1],
                   plotObj.grid[2],indexing='ij')
endrec= plotObj.maxrec-ha

x = plotObj.grid[0]
y = plotObj.grid[1]
z = plotObj.grid[2]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
dx = float(plotObj.header.binsize1)
dy = float(plotObj.header.binsize2)
dz = float(plotObj.header.binsize3)
CellVolume = dx*dy*dz


# Get data shape
data = plotObj.read(startrec=startrec, endrec=startrec)
mid = [int(data.shape[i]/2.)-1 for i in range(3)]
cnt = 0


gamma_tip = []
gamma_film = []
user_locations =  [0,1,2,4,8,12,16,24,30,40] 
strip_width = 10

gamma_locs = []
gamma_locs_sd = []


for user_location in user_locations:
    
    # this user_location is delta: distance from the tip

    for rec in range(startrec, endrec, skip):
        
        #First plot all the slices, then the radial averaged rotated Pressure
        fig, ax = plt.subplots(2,2,figsize=(15,15)) ## to plot contours of pressure profiles
        
        rave = []; Pave = []
        rho = rhoObj.read(startrec=rec, endrec=rec)
        rho = np.mean(rho[:,:,:,:,0],3)
        
        #Average around in a radial sweep
        for i in range(data.shape[0]): 
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
        xp= Xr[coordinates[0]+1, coordinates[1]+1]    # y coordinates for fitting 
        rp= R[coordinates[0]+1, coordinates[1]+1]    # z coordinates for fitting 
    
        #T ake R(t) and add fraction of domain #3*rp.max()/4.
        #####################################################
        rtop = rp.min() + rp.max()/3. 
        xp = xp[rp < rtop]; rp = rp[rp < rtop]
        #Plot a line around the interface by ordering the points by proximity
        #Think this is like the "alpha shape" algorithm
        r = order(xp, rp); Np = r.shape[0]
        ax[1,1].plot(r[:,0], r[:,1], 'k-')
        path = Path(r)
    
    
        data = plotObj.read(startrec=rec, endrec=rec)
        data = np.mean(data[:,:,:,:,:],3)
        PN, PT1, PT2 = transform_pressure_tensor(data[:,:,:,:], Y2, Z2)
        d = PT1
    
        # Average around in a radial sweep
        for i in range(data.shape[0]):
            Pave.append(radial_profile(d[i,:,:],[mid[1],mid[2]]))
        Pave = np.array(Pave)
    
        ax[0,0].pcolormesh(d[:,mid[1],:], cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
        ax[1,0].pcolormesh(d[:,:,mid[2]], cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
        ax[0,1].pcolormesh(d[mid[0],:,:], cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
        cm=ax[1,1].pcolormesh(Xr, R, Pave, cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
        #plt.colorbar(cm)
        plt.show()
    
    
        # Next get the surface tension estimate
        P = {}
        for n, d in zip(["PN", "PT1", "PT2"], [PN, PT1, PT2]):
    
            #Average around in a radial sweep
            Pave = []
            for i in range(data.shape[0]):
                Pave.append(radial_profile(d[i,:,:],[mid[1],mid[2]]))
            P[n] = np.array(Pave)
    
        
        #Plot contour, pressure and integral
        fig, ax = plt.subplots(3,2, sharex=True, figsize=(15, 15))
        
    
        for i in range(2):
            if i==0:
                #Bottom of film, delta = 0
                topindx = int(rp.min()/dR) 
                #s = topindx; e = topindx+10
                s = topindx 
                e = s + strip_width # originally +10
                tipLoc = s
            else:
                #delta = user_location
                #s = topindx-5; e = topindx+5
                topindx = int(rp.min()/dR)
                s = topindx + user_location  
                e = s + strip_width 
    
            # tangential pressure
            ax[0,i].pcolormesh(Xr, R, 0.5*(P["PT1"]+P["PT2"]), cmap=plt.cm.RdYlBu_r)
            ax[0,i].plot(Xr[:,s],R[:,s], 'k--')
            ax[0,i].plot(Xr[:,e],R[:,e], 'k--')
            ax[0,i].set_ylabel(r"$r$")
            
            
            ax[1,i].plot(Xr[:,0],np.mean(P["PN"][:,s:e],1), label="$P_N$")
            ax[1,i].plot(Xr[:,0],np.mean(P["PT1"][:,s:e],1), label="$P_{T2}$")
            ax[1,i].plot(Xr[:,0],np.mean(P["PT2"][:,s:e],1), label="$P_{T1}$")
            ax[1,i].legend(fontsize=10, loc=1)
            
            if i==0:                
                ax[1,i].set_ylabel(r"$P$")
            #ax[1].set_xlabel(r"$x$")
            
            PNmPT = (np.mean(P["PN"][:,s:e],1) 
                       - 0.5*(np.mean(P["PT1"][:,s:e],1)+np.mean(P["PT2"][:,s:e],1)))
            
            ax[2,i].fill_between(Xr[:,0], 0., PNmPT, alpha=0.5)
            ax[2,i].plot(Xr[:,0], PNmPT)
            
            #Define range of integration for Kirkwood Buff
            startint = 0
            endint = PNmPT.shape[0]-1
            ax[2,i].vlines(Xr[startint,0], PNmPT.min(), PNmPT.max())
            ax[2,i].vlines(Xr[endint,0], PNmPT.min(), PNmPT.max())
            gamma = np.zeros(PNmPT.shape[0])
            
            for n in range(startint,endint):
                gamma[n] = 0.5*np.trapz(PNmPT[startint:n], dx=dx)
                
            if i ==0:
                gamma_tip.append(np.round(gamma[endint-1],4))
            else:
                gamma_film.append(np.round(gamma[endint-1],4))
                
                
                
            ax2 = ax[2,i].twinx()
            ax2.plot(Xr[:,0], gamma, "r")
            ax[2,i].set_xlabel(r"$x$")
            if i==0:
                ax[2,i].set_ylabel(r"$P_N - P_T$")
            else:
                ax2.set_ylabel(r"$\gamma$", color="r")
                
            ax[2,i].text(Xr[int(0.5*(startint+endint)),0], 
                             0.5*(PNmPT.max()+PNmPT.min()), 
                      "$\gamma = $" + str(np.round(gamma[endint-1],4) ))
            
        plt.show()
    
 
    import scipy.stats as st
    g_tip = np.asarray (gamma_tip)
    g_tip_mean = np.mean (gamma_tip)
    g_tip_sd = np.std (gamma_tip)
    g_film = np.asarray(gamma_film)
    g_film_mean = np.mean (gamma_film)
    g_film_sd = np.std (gamma_film)
    g_film_sem = st.sem(gamma_film)
    

    gamma_locs.append(g_film_mean)
    gamma_locs_sd.append(g_film_sd)

gamma_max = (np.asarray(gamma_locs)).max()
r_max = r.max()
R_max = R.max()

plt.figure(figsize=(6,8))
plt.errorbar(np.asarray(gamma_locs)/gamma_max, (np.asarray(user_locations))*dy, xerr=g_film_sem, marker='o', mfc='red',
         mec='k', ms=20, mew=3, color='k', linestyle='--', ecolor='k', elinewidth=2, capsize=2)

plt.xlim([0.8,1.05])
plt.xlabel(r'$\gamma (r)/\gamma$')
plt.ylabel(r'$r$')
