#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:01:24 2022
@author: muhammadrizwanurrahman
@author: Additions 06/11/22 by edwardsmith999 to fit ellipse+transform
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import griddata

def reciprical(x, c):
    assert c.shape[0] == 2
    return c[0]/x + c[1] 

def reciprical_power(x, c):
    assert c.shape[0] == 3
    return c[0]/x**c[1] + c[2] 

def quadratic(x, c):
    assert c.shape[0] == 3
    return c[0]*x**2 + c[1]*x + c[2] 

def d_quadratic_dx(x, c):
    assert c.shape[0] == 3
    return 2.*c[0]*x + c[1]

def cubic(x, c):
    assert c.shape[0] == 4
    return c[0]*x**3 + c[1]*x**2 + c[2]*x + c[3] 

def quartic(x, c):
    assert c.shape[0] == 5
    return c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + c[3]*x + c[4] 

def cosh(x, c):
    assert c.shape[0] == 4
    return c[2] - c[0]*(1.-np.cosh(c[1]*(x-c[3])))

def fit_reciprical_power(x1, y1, x2, y2, df1dx):
    c = np.zeros(3)
    a = (x1*(y1-y2))
    c[1] = -df1dx/a
    c[0] = a*x1**c[1]
    c[2] = y2
    return c

def fit_reciprical(x1, y1, x2, y2):
    A = np.array([[1./x1, 1],
                  [1./x2, 1]])
    b = np.array([y1, y2])
    return np.linalg.solve(A, b)

def fit_quadratic(x1, y1, x2, y2, df1dx):
    A = np.array([[x1**2, x1, 1],
                  [x2**2, x2, 1],
                  [2*x2, 1, 0]])
    b = np.array([y1, y2, df1dx])
    return np.linalg.solve(A, b)

def fit_cubic(x1, y1, x2, y2, df1dx, df2dx):
    A = np.array([[x1**3, x1**2, x1, 1],
                  [x2**3, x2**2, x2, 1],
                  [3*x1**2, 2*x1, 1, 0],
                  [3*x2**2, 2*x2, 1, 0]])
    b = np.array([y1, y2, df1dx, df2dx])
    return np.linalg.solve(A, b)

def fit_quartic(x1, y1, x2, y2, df1dx, df2dx, d2f1dx2):
    A = np.array([[x1**4, x1**3, x1**2, x1, 1],
                  [x2**4, x2**3, x2**2, x2, 1],
                  [4*x1**3, 3*x1**2, 2*x1, 1, 0],
                  [4*x2**3, 3*x2**2, 2*x2, 1, 0],
                  [12*x2**2, 6*x2, 2, 0, 0]])
    b = np.array([y1, y2, df1dx, df2dx, d2f1dx2])
    return np.linalg.solve(A, b)

def fit_cosh(x1, y1, x2, y2, df1dx, 
             coeff = np.array([1.,1.,0.,0.]),
             Maxiter=2000, tol=1e-4):
    #Intial guess for coefficients b and c
    #where we iterate as this is a Newton solver
    b = coeff[0]
    c = coeff[1]
    L = x1-x2
    for n in range(Maxiter):        
        f1 = y2 - b*(1. - np.cosh(c*L)) - y1
        f2 = b*c*np.sinh(c*L) - df1dx

        F = np.array([f1, f2])
        df1db = 1. - np.cosh(c*L)
        df1dc = b*L*np.sinh(c*L)
        df2db = c*np.sinh(c*L)
        df2dc = b*np.sinh(c*L) + b*c*L*np.cosh(c*L)

        J = np.array([[df1db,df1dc],
                      [df2db,df2dc]])

        diff_xn = np.linalg.solve(J, F)

        #Try a new root if diverging
        if (np.sum(np.abs(diff_xn)) > 1e2):
            b, c = (np.random.rand(2)-0.5)
            diff_xn = 0.1

        xn = np.array([b, c])
        xn -= diff_xn
        #print("iter=", n, "diff_xn", diff_xn, "b,c", xn)#, "F", F, "J", J)
        if (np.sum(np.abs(diff_xn)) < tol):
            break
        b = xn[0]
        c = xn[1]
        
    coeff[0] = xn[0]
    coeff[1] = xn[1]
    coeff[2] = y2
    coeff[3] = x2
    return coeff


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

    points = list(zip(xp, zp))
    indx = np.argmax(zp)
    if xp[indx] < 0:
        startindx = indx
    else:
        startindx = 0
    opoints = OrderByDistance(points, startindx)
    return np.array(opoints)


def find_neck_position_V2(H,R):
    # H is defines as => r[:,0][mid:-1]
    # R is defined as => r[:,1][mid:-1]
    
    start = 1 # discarding first data point
    gr = (np.gradient(H[start:])/np.gradient(R[start:])) 
        
    tol = 0.01  
    ## if 10 consequitive derivatives are zero - we take it as a 
    ## confirmation of the flat unperturbed film
    
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
            xneck = H[j-1]
            neck_idx = j-1
            break 
        else:
            neck = np.nan
            xneck = np.nan
            neck_idx = np.nan 

    return neck, xneck, neck_idx

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile  

def cart2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2ellipse(X, Y, a):
    
    z = X + Y*1j; 
    el = np.arccosh(z/a);
    mu = el.real;
    nu = el.imag;
    return mu, nu

def ellipse_transform_pressure_tensor(Pxx, Pxy, Pyy, X, Y, w, h):

    #Convert position of ith and jth cell of pVA[i,j,:]
    #to its mu and nu values, using X, Y meshgrid
    a = np.sqrt(w**2 - h**2)
    mu, nu = cart2ellipse(X, Y, a)

    J = np.array([[np.sinh(mu)*np.cos(nu), -np.cosh(mu)*np.sin(nu)],
                  [np.cosh(mu)*np.sin(nu), np.sinh(mu)*np.cos(nu)]])
    if (a> 1e-6):
        Jdet = a**2 * (np.cosh(mu)**2-np.cos(nu)**2)
        J = J * a**2 / Jdet
    P = np.array([[Pxx, Pxy],
                  [Pxy, Pyy]])
    JP = np.einsum("il...,lj...->ij...", J, P)
    JPJ = np.einsum("il...,jl...->ij...", JP, J)
    PN = JPJ[0,0]
    PT = JPJ[1,1]

    return mu, nu, PN, PT

def ellipse(x, dw2, dh2):
    return (dh2*(1.-x**2/dw2))**(0.5)

def D_ellipseDt(x, dw2, dh2):

    return -(np.sqrt(dh2)/(dw2))*x/((1.-x**2/dw2)**(.5))

def transform_velocity(u, X, Y):
    ux = u[...,0] 
    uy = u[...,1] 

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
    PT2 = ( Pxx*np.sin(theta)**2 
          + Pyy*np.cos(theta)**2  
          - 2.*Pxy*np.sin(theta)*np.cos(theta))

    return PN, PT1, PT2


def J_transform_pressure_tensor(P, X, Y):
    Pxx = P[...,0]
    Pxy = P[...,1] 
    Pyy = P[...,2] 

    #Convert position of ith and jth cell of pVA[i,j,:]
    #to its theta value, using X, Y meshgrid
    r, theta = cart2polar(X, Y)

    J = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    P = np.array([[Pxx, Pxy],
                  [Pxy, Pyy]])
    JP = np.einsum("il...,lj...->ij...", J, P)
    PT = np.einsum("il...,jl...->ij...", JP, J)
    PT1 = PT[0,0]
    PT2 = PT[1,1]
    #assert(PT[0,1] == PT[1,0])
    PNT = PT[0,1]
    return PT1, PT2, PNT


def normal_transform_pressure_tensor(P, dydx):
    Pxx = P[...,0]
    Pxy = P[...,1]
    Pyy = P[...,2]

    #Convert position of ith and jth cell of pVA[i,j,:]
    #to its theta value, using X, Y meshgrid
    theta = np.arctan(1./dydx)

    J = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    P = np.array([[Pxx, Pxy],
                  [Pxy, Pyy]])
    JP = np.einsum("il...,lj...->ij...", J, P)
    PT = np.einsum("il...,jl...->ij...", JP, J)
    PT1 = PT[0,0]
    PT2 = PT[1,1]
    PNT = PT[0,1]
    return PT1, PT2, PNT

def get_norm_tang_line(x, y, dydx, normdydx, xtangnt, norm=False):

    xnorm = np.copy(xtangnt)
    ytangnt = dydx*xtangnt
    ynorm = normdydx*xtangnt

    if norm:
        lngthx = xtangnt.max() - xtangnt.min()
        lngthy = ytangnt.max() - ytangnt.min()
        mag = np.sqrt(lngthx**2 + lngthy**2)
        xtangnt = norm*xtangnt/mag
        ytangnt = norm*ytangnt/mag
        lngthy = ynorm.max() - ynorm.min()
        mag = np.sqrt(lngthx**2 + lngthy**2)
        xnorm = norm*xnorm/mag
        ynorm = norm*ynorm/mag

    xt = x+xtangnt
    yt = y+ytangnt

    xn = x+xnorm
    yn = y+ynorm

    return xn, yn, xt, yt

def get_norm_tang_fn(x, fn, dfn, c, xtangnt, norm=False):

    y = fn(x, c)
    dydx = dfn(x, c)

    if np.abs(dydx) > 1e-5:
        normdydx = -1./dydx 
    else:
        normdydx = 1e5

    xn, yn, xt, yt = get_norm_tang_line(x, y, dydx, normdydx, xtangnt, norm)

    return xn, yn, xt, yt, normdydx, dydx

def get_norm_tang_ellipse(x, dw2, dh2, xtangnt, norm=False):

    y = ellipse(x, dw2, dh2)
    dydx = D_ellipseDt(x, dw2, dh2)

    if np.abs(dydx) > 1e-5:
        normdydx = -1./dydx 
    else:
        normdydx = 1e5

    if normdydx <= 0.:
        xtangnt = -xtangnt

    xn, yn, xt, yt = get_norm_tang_line(x, y, dydx, normdydx, xtangnt, norm)


    return xn, yn, xt, yt, normdydx, dydx


ppdir = './postproclib/'
sys.path.append(ppdir)
import postproclib as ppl


fdir = './results/'
skip = 5
ha = 5
startrec = 80 + ha

### ................................................................ #####

  
PPObj = ppl.All_PostProc(fdir)
print(PPObj)
#Get plotting object
rhoObj = PPObj.plotlist['rho']
try:
    plotObj = PPObj.plotlist['pVA']
    incstress = True
except KeyError:
    incstress = False

x, y, z = rhoObj.grid
Y,Z = np.meshgrid(y,z)

 
dx = float(rhoObj.header.binsize1)
dy = float(rhoObj.header.binsize2)
dz = float(rhoObj.header.binsize3)
CellVolume = dx*dy*dz 
tplot = float(rhoObj.header.tplot)
mass_avg_steps = float(rhoObj.header.Nmass_ave)
deltat = float(rhoObj.header.delta_t)
totalSteps = float(rhoObj.header.Nsteps)
endrec = rhoObj.maxrec - ha


## declaring empty arrays to store data in the loop
l_aureole_store = []
alpha_store = []
a_store = []
b_store = []

for rec in range(startrec, endrec, skip):

    
    #fig,ax = plt.subplots(figsize=(8,8))
    rhodata = rhoObj.read(startrec=rec-ha, endrec=rec+ha)

    mid = [int(rhodata.shape[i]/2.)-1 for i in range(3)] 
    
    ## Radial profile 
    rave = []    
    rho = np.mean(rhodata[:,:,:,:,0],3)
    for i in range(rhodata.shape[0]):
        rave.append(radial_profile(rho[i,:,:],[mid[1],mid[2]]))
    rave = np.array(rave)

    # Load Pressure profile
    if incstress:
        data = plotObj.read(startrec=rec, endrec=rec)
        data = np.mean(data[:,:,:,:,:],3)

        PN = data[:,:,:,0]
        Prr, Ptt, Prt = J_transform_pressure_tensor(data[:,:,:,[4,5,8]], Y, Z)

#        fig, ax = plt.subplots(2,2)
#        ax[0,0].pcolormesh(PN[:,mid[1],:], cmap=plt.cm.RdYlBu_r)
#        ax[1,0].pcolormesh(Prr[:,mid[1],:], cmap=plt.cm.RdYlBu_r)
#        ax[0,1].pcolormesh(Ptt[:,mid[1],:], cmap=plt.cm.RdYlBu_r)
#        ax[1,1].pcolormesh(Prt[:,mid[1],:], cmap=plt.cm.RdYlBu_r)
#        plt.show()

        #Average around in a radial sweep
        PTrans = np.zeros([rave.shape[0],rave.shape[1],4])
        for n, d in enumerate([PN, Prt, Prr, Ptt]):
            Pave = []
            for i in range(data.shape[0]):
                Pave.append(radial_profile(d[i,:,:],[mid[1],mid[2]]))
            PTrans[:,:,n] = np.array(Pave)

#        fig, ax = plt.subplots(2,2)
#        ax[0,0].pcolormesh(PTrans[:,:,2], cmap=plt.cm.RdYlBu_r, vmin=-0.2, vmax=0.2)
#        ax[1,0].pcolormesh(PTrans[:,:,0], cmap=plt.cm.RdYlBu_r, vmin=-0.2, vmax=0.2)
#        ax[0,1].pcolormesh(PTrans[:,:,1], cmap=plt.cm.RdYlBu_r, vmin=-0.2, vmax=0.2)
#        ax[1,1].pcolormesh(PTrans[:,:,3], cmap=plt.cm.RdYlBu_r, vmin=-0.2, vmax=0.2)
#        plt.show()

        #Use two tangents (P theta theta and Prr) to improve stats
        PTrans[:,:,2] = 0.5*(PTrans[:,:,2] + PTrans[:,:,3])

        #Test values to check rotation
        #PTrans[:,:,0] = 1.
        #PTrans[:,:,1] = 0.
        #PTrans[:,:,2] = 0.

    maxr = np.sqrt(y.max()**2 + z.max()**2)
    rad = np.linspace(0.,maxr,rave.shape[1])
    dR = np.diff(rad)[0]
    Xr, R = np.meshgrid(x, rad, indexing="ij")
    
    ## Get outer surface coordinates
    rhocutoff = 0.4
    mask = rave > rhocutoff
    difvalue=(np.abs(np.diff(np.array(mask,dtype=int),axis=0))[:,:-1]+ 
              np.abs(np.diff(np.array(mask,dtype=int),axis=1))[:-1,:]) > 0.
    coordinates=np.where(difvalue==True)
    
    #Add first value to end so it loops around
    xp=Xr[coordinates[0]+1, coordinates[1]+1]    # y coordinates for fitting 
    rp= R[coordinates[0]+1, coordinates[1]+1]    # z coordinates for fitting 
    r = order(xp, rp); Np = r.shape[0]
    d = np.mean(rhodata[...,0],3)

    ## find blob height, blob width, maximum_height location, neck position, aureole length
    rl = r.shape[0]//2 # take only half along r dir  about the axis of symmetry     
    xdir = r[rl:-1,1] # direction of retraction
    rdir = r[rl:-1,0] # direction of hole growth
    
    """
    ^r  _--_
    |  .    ----------------->
    |  .     
    ............................ axis of retraction, x, also (...) is the symmytry line
    
    
    """
    
    rmax = rdir.max() # maximum half-height is euqal to the maximum ordinate 
    rmax_idx = int(np.mean(np.where(rdir==rmax))) # idx of max height
    loc_rmax = xdir[rmax_idx] # location of max height is also the location of the center of the ellipse
    half_blob_width_in_rdir = loc_rmax - rp.min() # width of the ellipse = dist. of the ellipse center - dist. of the ellipse tip
    loc_ellipse_end = loc_rmax + half_blob_width_in_rdir  
    
    rmid = len(r[:,0])//2
    neck_pos, xneck_pos, neck_idx = find_neck_position_V2(r[:,0][rmid:-1],r[:,1][rmid:-1])
    l_aureole = np.abs(loc_ellipse_end-neck_pos)
    

    #ELLIPSE MAPPING
    rm = rp.min(); 
    dw = half_blob_width_in_rdir; dh = rmax;
    dw2 = dw**2; dh2 = dh**2
    #print("dw=", dw, "dh=", dh, "loc_rmax", loc_rmax)
    xcenter = 0. # xp.max() + xp.min()
    rcenter = loc_rmax
    rep = np.linspace(-dw, dw, 100)
    
    #Shift everything to ellipse centre at zero
    R -= rcenter
    r -= rcenter
    rad -= rcenter
    rp -= rcenter
    loc_ellipse_end -= rcenter
    neck_pos -= rcenter
    loc_rmax -= rcenter
    rdir += rcenter
    points = np.zeros([Xr.size,2])
    points[:,0] = Xr.ravel()
    points[:,1] = R.ravel()

    #Plot location of joining line
    if incstress:

        vmn = None#-0.2
        vmx = None#0.2
    else:
        qnty = rave
        vmn = 0.0#None#-0.2
        vmx = 0.8 #None#  0.05

    ## show instantaneous detection of the neck and aureole
    fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,7))
    axr = ax[0]

    #Plot colormesh and show all ellipses, necks, etc
    cm1 = axr.pcolormesh(Xr, R, rave, cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
    plt.colorbar(cm1, ax=axr)
    
    #plt.axis('off')

    #if incstress:
    #    axr.quiver(Xr, R, PTrans[:,0], PTrans[:,2], scale=25.) 
    ms=16
    axr.plot(rdir,xdir,'ko',ms=ms)
    axr.plot(rmax,loc_rmax,'ro',ms = ms)
    axr.plot(0,loc_ellipse_end,'bo',ms = ms)
    axr.plot(np.linspace(Xr.min(), Xr.max(),10), neck_pos*np.ones(10),'m-')
    axr.plot(xneck_pos, neck_pos, 'mo',ms=ms)

    axr.plot( ellipse(rep, dw2, dh2), rep, 'k-')
    axr.plot(-ellipse(rep, dw2, dh2), rep, 'k-')
    
    
    #The index of the ellipse to join to neck
    startindx = 50; endindx = 90; all_diff = []
    #Loop over ellipse and see when two gradients are closest
    for joinindx in range(startindx,endindx):
        coeff = fit_quadratic(rep[joinindx], ellipse(rep[joinindx], dw2, dh2), neck_pos, xneck_pos, 0.)
        grad_diff = D_ellipseDt(rep[joinindx], dw2, dh2)-d_quadratic_dx(rep[joinindx], coeff)
        all_diff.append(grad_diff)
    minindx = np.argmin(np.abs(np.array(all_diff)))
    joinindx=startindx + minindx
    coeff = fit_quadratic(rep[joinindx], ellipse(rep[joinindx], dw2, dh2), neck_pos, xneck_pos, 0.)
    #print("Gradients at join", joinindx, rep[joinindx], 
    #      D_ellipseDt(rep[joinindx], dw2, dh2), 
    #      d_quadratic_dx(rep[joinindx], coeff))

    #Plot join to ellipse
    ellipse_join = ellipse(rep[joinindx], dw2, dh2)
    axr.plot(ellipse_join, rep[joinindx],'yo')

    #Define line parameters
    skip = 1; pskip=10
    normmin = -3; normmax = 3; Np=40; lngth = 20. #normmax-normmin
    xtangnt = np.linspace(normmin, normmax, Np)
    along_line = []; line_pos = []
    for pointno, re in enumerate(rep[0:joinindx:skip]):
        ye = ellipse(re, dw2, dh2) 

        #Plot tangent at intercept
        xn, yn, xt, yt, normdydx, dydx = get_norm_tang_ellipse(re, dw2, dh2, xtangnt, lngth)

        if (pointno%pskip == 0):
            axr.plot(ye, re, 'go', ms=1)
            axr.plot(yn, xn,  'g-', lw=0.5)

        #If we want to rotate a qnty, then can do it based on angle of normal
        if incstress:
            #print("Ellipse=", re, ye, normdydx, np.arctan(1./normdydx)/np.pi*180)
            PTrans_N, PTrans_T, PTrans_NT = normal_transform_pressure_tensor(PTrans, normdydx)
            qnty = PTrans_N - PTrans_T

        #line_sample = griddata(points, qnty.ravel(), (yn, xn), method='linear')
        line_sample = griddata(points, qnty.ravel(), (yn, xn), method='nearest')
        
        line_pos.append(re)
        along_line.append(line_sample)

    re = rep[joinindx]
    ye = ellipse(re, dw2, dh2) 

    #Draw line connecting ellipse to  neck
    xneck = np.linspace(re, neck_pos, 100)
    coeff = fit_quadratic(re, ye, neck_pos, xneck_pos, 0.)
    axr.plot(quadratic(xneck, coeff), xneck, 'k-', lw=2,  label='quadratic')

    #Draw tangents and normals to neck
    fn = quadratic; dfn = d_quadratic_dx
    for pointno, re in enumerate(xneck[::skip]):

        ye = fn(re, coeff)
        xn, yn, xt, yt, normdydx, dydx = get_norm_tang_fn(re, fn, dfn, coeff, xtangnt, lngth)

        #Plot tangent at intercept
        if (pointno%pskip == 0):
            axr.plot(ye, re, 'go', ms=1)
            #axr.plot(yt, xt, 'g-', lw=0.5)
            axr.plot(yn, xn,  'g-', lw=0.5)

        #If we want to rotate a qnty, then can do it based on angle of normal
        if incstress:
            #print("Line=", re, ye, normdydx, np.arctan(1./normdydx)/np.pi*180)
            PTrans_N, PTrans_T, PTrans_NT = normal_transform_pressure_tensor(PTrans, normdydx)
            qnty = PTrans_N - PTrans_T

        #line_sample = griddata(points, qnty.ravel(), (yn, xn), method='linear')
        line_sample = griddata(points, qnty.ravel(), (yn, xn), method='nearest')
        
        #print("line=", pointno, re, xn, yn, line_sample)
        line_pos.append(re)
        along_line.append(line_sample)

    #Plot the unrolled stress field
    number = np.array(line_pos) #np.arange(len(along_line))
    a, b = np.meshgrid(number, xtangnt, indexing="ij")
    #cm2 = ax[1].pcolormesh(b, a, np.array(along_line), cmap=plt.cm.RdYlBu_r, vmin=vmn, vmax=vmx)
    cm2 = ax[1].pcolormesh(b, a, np.array(along_line), cmap=plt.cm.RdYlBu_r, vmin=-0.1, vmax=0.15) #MRR

    plt.colorbar(cm2, ax=ax[1])
    ax[1].plot(0.,loc_rmax,'ro',ms=ms)
    ax[1].plot(0.,rep[joinindx],'yo',ms=ms)
    
    
    ax[2].plot(0.5*np.trapz(PTrans[:,:,0]-PTrans[:,:,2], dx=dx, axis=0), rad, 'o',ms = 14, mfc = 'r')
    ax[2].plot(0.5*np.trapz(np.array(along_line), dx=dx, axis=1), np.array(line_pos), 'o', ms = 14, mfc='b',mec='k',markevery = 10)
    #ax[2].plot(np.trapz(np.array(along_line), dx=dx, axis=1), np.array(line_pos), 'bo')

    axr.set_xlim([-20,20])
    #axr.set_ylim([rp.min()-10,neck_pos+10]) 
    axr.set_ylim([-20,150]) # MRR
    ax[2].set_xlim([-0.5,1.5])
    

    
    #axr.set_aspect('equal', 'box')
    #plt.yticks([])
    plt.show()
    
    """
    ### separately draw the surface tension plot (mapped and un-mapped)
    f,ax3 = plt.subplots(figsize = (5,10))
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.size'] = 40
    plt.rcParams['axes.labelsize'] = 38
    plt.rcParams['font.family'] = 'Times New Roman' #'sans-serif'
    plt.rcParams['axes.linewidth'] = 3 #set the value globally
    plt.rcParams['lines.linewidth'] = 3 #set the value globally
    ax3.plot(0.5*np.trapz(PTrans[:,:,0]-PTrans[:,:,2], dx=dx, axis=0), rad, 'o',ms = 14, mfc = 'r', markevery = 2)
    ax3.plot(0.5*np.trapz(np.array(along_line), dx=dx, axis=1), np.array(line_pos), 'o', ms = 14, mfc='b',mec='k', markevery = 4)
    ax3.set_xlim([-0.5,1.5])
    plt.ylim([-50,150])
    plt.title(str(rec))
    plt.tick_params(axis='both', which='major', pad=15, length=8, width=2)
    plt.show()
    """
