#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:48:31 2022
@author: muhammadrizwanurrahman
"""

import os
import sys
from Func_hole_growth import get_hole_growth_p2p
import numpy as np


### ............................................................................
rec0 = 0 #  initial time
endrec = 1000 # change dependeing on the length of simulation
skip = 5
ha = 3
### ............................................................................

# saving directory
sdir = './savedata/'

fdir0 =  'film-rupture-h1/'
fdirs = ['ensemble_01/','ensemble_02/','ensemble_03/']

# there are two algorithms for tracking the growth,
algo_radial = False
algo_p2p = True

for f in fdirs:
    fdir = fdir0 + f + 'results/'
    
    growthdata = get_hole_growth_p2p (fdir,rec0,skip,ha,showfig = False)
    # output data contains: [mdtime, hole_rad, hole_num, h0_film]
    time = growthdata[0]
    R = growthdata[1]
    num = growthdata [2]
    h0 = growthdata[3]
    savedata = [time,R,num,h0]
        
    fname = sdir + f[0:-1] + '.npy'

    savedata = np.asarray(savedata)
    np.save(fname,savedata)
    
