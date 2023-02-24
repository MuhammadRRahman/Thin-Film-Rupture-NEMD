#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 06:35:22 2022

@author: Muhammad R Rahman, Edward R Smith
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

def cart2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def Closest(p, points):
    c = np.array(p) #2 elements
    a = np.array(points) #N by 2 elements
    dist = (a[:,0]-c[0])**2 + (a[:,1]-c[1])**2
    indx = np.argsort(dist)
    return indx[0]

def OrderByDistance(points, startindx=0):
    points = list(points)
    start = points[startindx]
    current = start
    remaining = points
    path = [start];
    for i in range(100000):
        nextindx = Closest(current, remaining);
        current = remaining[nextindx]
        path.append(current)
        remaining.pop(nextindx)
        if (remaining == []):
            break
    return path;

def order(xp, zp):

    #First order one of the coordinates
    #indx = np.argsort(xp)
    #xp = xp[indx]
    #zp = zp[indx]

    #Then order by distance and return numpy array
    points = list(zip(xp, zp))
    indx = np.argmax(zp)
    if xp[indx] < 0:
        startindx = indx
    else:
        startindx = 0
    opoints = OrderByDistance(points, startindx)
    return np.array(opoints)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile    


def get_normals(r):
    dx = 0.5*(r[1:,0] - r[:-1,0])
    dy = 0.5*(r[1:,1] - r[:-1,1])
    dx = np.append(dx, 0.5*(r[0,0]-r[-1,0]))
    dy = np.append(dy, 0.5*(r[0,1]-r[-1,1]))
    midpoints = np.zeros([dx.shape[0],2])
    midpoints[:,0] = r[:,0]+dx
    midpoints[:,1] = r[:,1]+dy
    norms = np.zeros([dx.shape[0],2])
    norms[:,0] = -dy
    norms[:,1] = dx
    return midpoints, norms


def transform_velocity(ux, uy, X, Y):


    #Convert position of ith and jth cell of pVA[i,j,:]
    #to its theta value, using X, Y meshgrid
    r, theta = cart2polar(X,Y)

    #Convert each value
    ur = ux*np.cos(theta) + uy*np.sin(theta)
    ut = (1/r)*(uy*np.cos(theta) - ux*np.sin(theta))

    return ur, ut

 

def transform_pressure_tensor(P, X, Y):
    PN  = P[...,0]
    Pxx = P[...,4]
    Pxy = P[...,5] 
    Pyy = P[...,8] 

    #Convert position of ith and jth cell of pVA[i,j,:]
    #to its theta value, using X, Y meshgrid
    r, theta = cart2polar(X, Y)

    #Convert each value
    PT1 = (  Pxx*np.cos(theta)**2 
          + Pyy*np.sin(theta)**2 
          + Pxy*np.sin(2.*theta))
    #PT = (  Pxx*np.cos(theta + 0.5*np.pi)**2 
    #      + Pyy*np.sin(theta + 0.5*np.pi)**2  
    #      + Pxy*np.sin(2.*theta+np.pi)      )
    PT2 = ( Pxx*np.sin(theta)**2 
          + Pyy*np.cos(theta)**2  
          - 2.*Pxy*np.sin(theta)*np.cos(theta))

    #t = np.random.rand(1)*np.pi
    #print(PT[0,0,0], PTa[0,0,0], t, np.cos(t + 0.5*np.pi)**2, np.sin(t)**2,np.cos(2.*t), np.sin(2.*t+np.pi))

    return PN, PT1, PT2


def find_neck_position_V2(H,R):
    #H = r[:,0][mid:-1]
    #R = r[:,1][mid:-1]
    start = 1 # discarding first data point
        
    gr = (np.gradient(H[start:])/np.gradient(R[start:]))
    tol = 0.01    
    for j in np.arange(0,len(gr)-10):
        
        t1 = abs(gr[j+0]  - gr[j+1])
        t2 = abs(gr[j+1]  - gr[j+2])
        t3 = abs(gr[j+2]  - gr[j+3])
        t4 = abs(gr[j+3]  - gr[j+4])
        t5 = abs(gr[j+4]  - gr[j+5])
        t6 = abs(gr[j+5]  - gr[j+6])
        t7 = abs(gr[j+6]  - gr[j+7])
        t8 = abs(gr[j+7]  - gr[j+8])
        t9 = abs(gr[j+8]  - gr[j+9])
        t10 = abs(gr[j+9]  - gr[j+10])
        
        t_arr = np.array([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10])
        
        
        if np.all((t_arr<=tol)):
            neck = R[j-1]
            neck_idx = j-1
            break 
        else:
            neck = np.nan
            neck_idx = np.nan 

    return neck, neck_idx



def count_rho(rtop,rmin,Xr,xp,rp,R,rave,difvalue):
        
    if rtop>rmin:    
        
        xp = xp[rp < rtop]; rp = rp[rp < rtop]        
        #Plot a line around the interface by ordering the points by proximity
        #Think this is like the "alpha shape" algorithm
        r = order(xp, rp); #Np = r.shape[0]        
        
        #fig, ax = plt.subplots(figsize=(8,8))
        #cm = ax.pcolormesh(Xr, R, difvalue, cmap='binary', vmin=0, vmax=1)
        #ax.plot(r[:,0], r[:,1], 'r-',linewidth=5)
        #ax.set_aspect(0.25)
        #ax.set_ylim([0,200])
        #plt.show()
            
        path = Path(r)
        
        #Get sum of density inside CV around end of film
        sumrave = 0.0; count = 0
        for i in range(Xr.shape[0]):
            for j in range(Xr.shape[1]):
                inside = path.contains_point([Xr[i,j], R[i,j]])
                if inside:
                    #print(i, [Xr[i,j], R[i,j]])
                    plt.plot(Xr[i,j], R[i,j], 'g.')
                    count += 1
                    sumrave += rave[i,j]  
                    print(sumrave)
                    
         
        #Compare to expected M=rho R(t) h0 from film
        averho = sumrave/count # molsInside/(count*CellVolume)
        indx_top = np.argmax(rp)
        h0 = r[-1,0]-r[0,0]
        Rt = rp.min() - 0
        mass_th = round(averho* h0*Rt,3)
        print('\n rho*h0*Rt :', mass_th)
           
    else:
        sumrave = np.nan 
        
    return sumrave  
