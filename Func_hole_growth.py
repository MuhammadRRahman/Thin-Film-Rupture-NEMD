#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:31:09 2022
Process Raw flowmol output data to measure hole radius over time

@author: Muhammad R Rahman
get_hole_growth_p2p() subroutine is developed by Edward R Smith
"""

def get_hole_growth_radial (fdir,rec0,skip,ha,showfig = False):
    
    """
    INPUT ARGS: 
        file directory
        initial record number
        frequency of analysis
        number of consecutive records to average over
        showfig: if False, no rim profile plotted, if a floating number, this sets the aspect ratio
        
    RETURNS:
        mdtime: (array) time in MD units
        hole_rad: (array) radius of the hole (tip of the film)
        total_rad: (array) radius measured until the neck
        blob_width: (array) width of the blob = neck - tip (in radial deirection)
        blob_height: (array) height or thickness of the blob (in the direction of film thickness)
        film_thickness: (array) thickness of the film
    
    """

    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    
    ## ........................................................................... ##
    from Func_surface_functions import radial_profile, order
    from Func_surface_functions import find_neck_position_V2
    ## ........................................................................... ##
    
    ppdir = './postproclib/'
    sys.path.append(ppdir)
    import postproclib as ppl
    
    ## ........................................................................... ##    
    #Get Data    
    fname = 'rho' # field to analyse
    PPObj = ppl.All_PostProc(fdir)
    print(PPObj)    
        
    #Get plotting object
    plotObj = PPObj.plotlist[fname]
    x, y, z = plotObj.grid
    Y,Z = np.meshgrid(y,z)
    
    testrec = plotObj.maxrec - 2*ha
    fnameS = np.mean(plotObj.read(startrec=testrec-ha, endrec=testrec+ha),3)
    mid = [int(fnameS.shape[i]/2.)-1 for i in range(3)] # ad
    endrec = min(500,plotObj.maxrec-1)   
        
    tplot = float(plotObj.header.tplot)
    mass_avg_steps = float(plotObj.header.Nmass_ave)
    deltat = float(plotObj.header.delta_t)
         
    hole_rad = []
    total_rad = []
    blob_width = []  # in radial direction
    blob_height = [] # in x-direction 
    mdtime = []
    film_thickness = []
    
    shiftby = 1
    if (showfig):
        fig,ax = plt.subplots(figsize=(8,6))   
         
    for rec in np.arange(rec0,endrec,skip):
        
        rave = []
        fname = np.mean(plotObj.read(startrec=rec-ha, endrec=rec+ha),3)
            
        for i in range(fname.shape[0]):
                
            rave.append(radial_profile(np.mean(fname[i,:,:,:],2),[mid[1],mid[2]]))
    
        rave = np.array(rave)
                
        maxr = np.sqrt(y.max()**2+z.max()**2)
        rad = np.linspace(0.,maxr,rave.shape[1])
        Xr, R = np.meshgrid(x, rad, indexing="ij") 
                       
        #fig, ax = plt.subplots(figsize=(8,8))
        #cm = ax.pcolormesh(Xr, R, rave, cmap=plt.cm.RdYlBu_r, vmin=0, vmax=1)

        #Get outer surface coordinates
        rhocutoff = 0.4
        mask = rave > rhocutoff
        difvalue=(np.abs(np.diff(np.array(mask,dtype=int),axis=0))[:,:-1]+ 
                  np.abs(np.diff(np.array(mask,dtype=int),axis=1))[:-1,:]) > 0.
        coordinates=np.where(difvalue==True)
        
        #Add first value to end so it loops around
        xp= Xr[coordinates[0]+1, coordinates[1]+1]    # y coordinates for fitting 
        rp= R[coordinates[0]+1, coordinates[1]+1]    # z coordinates for fitting 

        
        ##Choose top of CV to be the distance from edge of opening R(t) to the neck                
        rfilm = order(xp, rp)
        
        if (showfig):
            #fig,ax = plt.subplots(figsize=(8,8))
            #ax.plot(40*shiftby+r0new,r1new,'r-',linewidth=1)
            ax.plot(40*shiftby + rfilm[:,0] , rfilm[:,1], 'k-',ms=3, linewidth=2)
            ax.set_ylim([0,300])
            ax.set_aspect(showfig)
            plt.axis('off')
            shiftby+=1
            
        
        hfilm = abs(rfilm[:,0][3] - rfilm[:,0][-3])
        hblob = abs(rfilm[:,0].max() - rfilm[:,0].min()) # blob height in x-direction
        
        #print('\n thickness is: ', round(h0,3))
        fmid = rfilm.shape[0]//2
        Hcv = rfilm[:,0][fmid:-1]
        Rcv = rfilm[:,1][fmid:-1]   
        
        neck_pos, neck_idx = find_neck_position_V2(Hcv,Rcv)

        film_tip = rp.min()  
        
        if rec == rec0 :
            film_tip_0 = film_tip # initial radius
            
        blob_w = neck_pos - film_tip  
        #rtop = neck_pos  ## rtop set to neck
        
        
        timeis = rec * tplot * deltat *  mass_avg_steps ## md time
        
        ## SAVE DATA 
        blob_width.append(blob_w) 
        blob_height.append(hblob)
        ## radius upto film tip
        hole_rad.append(film_tip - film_tip_0 )
        ## radius upto the neck
        total_rad.append(neck_pos - film_tip_0 )
        ## initial film thickness
        film_thickness.append(hfilm)
        
        mdtime.append(timeis)       

    outData = [mdtime, hole_rad, total_rad, blob_width, blob_height, film_thickness]
    outData = np.asarray(outData)
        
    return outData
        
        
        
""" ### ******************************************************************* ###
    PEAK to PEAK surfae detection Algorithm
""" ### ******************************************************************* ###


def detectHole(binaryImageData):
    
    """
    Parameters
    ----------
    binaryImageData : 
        Binarized array of an image.
    
    Returns
    -------
    holeArea: pixelated area of the largest hole.
    numOfholes: total number of detected holes

    """

    import numpy as np
    from scipy.ndimage import measurements

    image = binaryImageData
    image_matrix = np.asmatrix(image.copy())      
                        
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    
    lw, numOfholes = measurements.label(image_matrix,structure=s) 
    labelList = np.arange(lw.max()+1)
    area = measurements.sum(image_matrix, lw, labelList)
    areaIm = area[lw]           
    holeArea = areaIm.max() 
    # pixelated area - for cut and pokes cases, only one hole exists
    # for spontaneous case, the first one is the largest one, once
    # more than one hole appears, break the loop from main code
    
    return holeArea, numOfholes


def get_hole_growth_p2p (fdir,rec0,skip,ha,showfig = False):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import zoom
    import sys

    ppdir = './postproclib/'
    sys.path.append(ppdir)
    import postproclib as ppl

    PPObj = ppl.All_PostProc(fdir)
    print(PPObj)
    
    #Get plotting object
    fname = 'rho'
    plotObj = PPObj.plotlist[fname]
    x, y, z = plotObj.grid
    Y,Z = np.meshgrid(y,z)
    rec = plotObj.maxrec - ha - 1
    smoothfactor = 1
    vmin = 0; vmax = 1 ;

    # get necessary dimensions from header file
    dx = float(plotObj.header.binsize1)
    dy = float(plotObj.header.binsize2)
    dz = float(plotObj.header.binsize3)

    tplot = float(plotObj.header.tplot)
    mass_avg_steps = float(plotObj.header.Nmass_ave)
    deltat = float(plotObj.header.delta_t)

    # First check whether plotting the right thing ! 
    rho = np.mean(plotObj.read(startrec=rec-ha, endrec=rec+ha),3)
    densitycut = 0.4 
    liquid = np.array(rho[:,:,:,0] > densitycut,dtype=int)
    interface = np.gradient(liquid[:,:,:],axis=0)
    thickness = np.zeros([interface.shape[1], interface.shape[2]])
    for i in range(interface.shape[1]):
        for j in range(interface.shape[2]):
            #Should be 2 values for top and bottom surface
            indx = np.where(interface[:,i,j] != 0)[0]
            if len(indx) != 0:
                thickness[i,j] = dx * (indx[-1] - indx[0])
            else:
                thickness[i,j] = 0.

    # Main task starts here   
    # mid = [int(rho.shape[i]/2.)-1 for i in range(3)] 
    thickness = np.zeros([interface.shape[1], interface.shape[2]])
    
    # start the main task here
    hole_rad = []
    mdtime = []
    hole_num = []
    smoothedthickness = np.nan 
    h0_film = np.nan
    
    startpoint = rec0
    endrec = plotObj.maxrec-ha-1
    skip = skip
    
    for rec in range(startpoint,endrec,skip):

        rho = np.mean(plotObj.read(startrec=rec-ha, endrec=rec+ha),3)
        liquid = np.array(rho[:,:,:,0] > densitycut,dtype=int)
        interface = np.gradient(liquid[:,:,:],axis=0)
        
        for i in range(interface.shape[1]):
            for j in range(interface.shape[2]):
                #Should be 2 values for top and bottom surface
                indx = np.where(interface[:,i,j] != 0)[0]
                if len(indx) != 0:
                    thickness[i,j] = dx * (indx[-1] - indx[0])
                else:
                    thickness[i,j] = 0.            
               

        smoothedthickness = zoom(thickness, smoothfactor)
        
        if rec == rec0 :
            #film_thickness_initial, h0
            h0_film = round(smoothedthickness.mean(),2)
            
        if (showfig):

            plt.imshow(smoothedthickness,cmap='inferno',vmin=vmin, vmax=vmax)
            title = round(rec*tplot * deltat *  mass_avg_steps,3)
            plt.colorbar()
            plt.title(title)
            plt.show()
                 
        binarythickness = smoothedthickness.copy()
        ## binarize the thickness array, holecutoff is just separating 0s and 1s
        holecutoff = 0.5
        binarythickness[binarythickness<=holecutoff] = 1        
        binarythickness[binarythickness>1] = 0        
            
        ###############   DETECT holes   #################               
        holeAreaPx, numOfholes = detectHole(binarythickness)
                        
        dr =  0.5*(dy+dz) 
        holeRadiusMD = np.sqrt(holeAreaPx/np.pi) * dr
        mdTime = rec*tplot * deltat *  mass_avg_steps
        
        if (holeRadiusMD>0):
            
            hole_rad.append(holeRadiusMD)
            hole_num.append(numOfholes)        
            mdtime.append(mdTime)

            
    ## blobRadius is the radius of the primary/first hole
    outData = np.asarray([mdtime, hole_rad, hole_num, h0_film])

    return outData



