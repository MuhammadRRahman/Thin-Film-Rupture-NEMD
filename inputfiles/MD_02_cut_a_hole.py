#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Edward R Smith
"""

import numpy as np
import os
import struct

class final_state:

    def __init__(self, fname= "./final_state", tether_tags = [3,5,6,7,10], verbose=False):
        self.fname = fname
        self.tether_tags = tether_tags

        #Get filesize and read headersize
        self.size = os.path.getsize(fname)
        self.headersize = np.fromfile(fname, dtype=np.int64, offset=self.size-8)
        with open(fname, "rb") as f:
            f.seek(self.headersize[0])
            self.binaryheader = f.read()

        self.read_header(verbose=verbose)

    def read_header(self, verbose=False):

        #Assume 14 doubles and work out integers
        self.ndbls = 14; self.ntrlint=4
        self.noints = int((len(self.binaryheader) - self.ndbls*8)/4)-self.ntrlint
        self.fmtstr = str(self.noints) + "i"+str(self.ndbls) +"d"+ str(self.ntrlint) + "i"
        print(self.fmtstr)
        self.hdata = list(struct.unpack(self.fmtstr, self.binaryheader))
        self.htypes = ["globalnp", "initialunits1", 
                      "initialunits2", "initialunits3", 
                      "Nsteps", "tplot", "seed1", "seed2",
                      "periodic1", "periodic2", "periodic3",
                      "potential_flag","rtrue_flag","solvent_flag",
                      "nmonomers","npx","npy","npz"]

        nproc = int(self.hdata[15])*int(self.hdata[16])*int(self.hdata[17])
        self.nproc = nproc
        [self.htypes.append("procnp"+str(p)) for p in range(nproc)]
        [self.htypes.append("proctethernp"+str(p)) for p in range(nproc)]
        [self.htypes.append(i) for i in 
                        ["globaldomain1", "globaldomain2",
                        "globaldomain3", "density", "rcutoff",
                        "delta_t", "elapsedtime", "simtime", 
                        "k_c","R_0", "eps_pp", "eps_ps", 
                        "eps_ss", "delta_rneighbr",
                        "mie_potential","global_numbering",
                        "headerstart","fileend"]]

        self.headerDict = {}
        for i in range(len(self.hdata)):
            if verbose:
                print(i, self.htypes[i], self.hdata[i])
            self.headerDict[self.htypes[i]]=self.hdata[i]

        if verbose:
            for k, i in self.headerDict.items():
                print(k,i)

    def read_moldata(self):

        #Read the rest of the data
        data = np.fromfile(self.fname, dtype=np.double, count=int(self.headersize/8))

        #Allocate arrays
        h = self.headerDict
        N = h["globalnp"]#self.N
        self.tag = np.zeros(N)
        self.r = np.zeros([N,3])
        self.v = np.zeros([N,3])
        self.rtether = np.zeros([N,3])
        self.Ntethered = 0

        #Create arrays for molecular removal
        self.Nnew = N
        self.delmol = np.zeros(N)
        self.molecules_deleted=False

        if (h["rtrue_flag"]):
            self.rtrue = np.zeros([N,3])
        if (h["mie_potential"]):
            self.moltype = np.zeros(N)
        if (h["global_numbering"]):
            self.globnum = np.zeros(N)
        if (h["potential_flag"]):
            self.potdata = np.zeros([N,8])

        i = 0
        for n in range(N):
            self.tag[n] = data[i]; i += 1
            self.r[n,:] = data[i:i+3]; i += 3
            self.v[n,:] = data[i:i+3]; i += 3

            if (h["rtrue_flag"]):
                self.rtrue[n,:] = data[i:i+3]; i += 3
            if (self.tag[n] in self.tether_tags):
                self.rtether[n,:] = data[i:i+3]; i += 3
                self.Ntethered += 1
            if (h["mie_potential"]):
                self.moltype[n] = data[i]; i += 1
            if (h["global_numbering"]):
                self.globnum[n] = data[i]; i += 1
            if (h["potential_flag"]):
                self.potdata[n,:] = data[i:i+8]; i += 8

        return self.tag, self.r, self.v

    def plot_molecules(self, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.scatter(self.r[:,0], self.r[:,1], self.r[:,2], c=self.tag[:])

    def remove_molecules(self, rpos, radius, rdim=0, ends=None, targetdensity=0.):
        
        """
        This method is designed to remove molecules from the simulation within a specified region, effectively creating a "hole" in the system. The parameters are:
        - `rpos`: The position of the center of the region where molecules will be removed.
        - `radius`: The radius of the region where molecules will be removed.
        - `rdim`: The direction of the region where molecules will be removed (0 for x, 1 for y, 2 for z).
        - `ends`: The ends of the region where molecules will be removed. If `ends` is `None`, the region extends infinitely in the `rdim` direction.
        - `targetdensity`: The target density of molecules in the region after removal. If `targetdensity` is greater than the current density, no molecules will be removed.
        
        The method first calculates the volume of the region where molecules will be removed, either as a cylinder (if `rdim` is not 3) or as a sphere (if `rdim` is 3).
        Then, it calculates the current density of molecules in the region. If the current density is less than or equal to the target density, no molecules are removed.
        If the current density is greater than the target density, the method calculates the ratio of the target density to the current density. This ratio is used as a 
        probability to decide whether each molecule in the region should be removed. This then loops over all molecules in the system. For each molecule, 
        it calculates its distance from the center of the region. If the molecule is within the region and its removal would not reduce the density below the target density,
        the molecule is marked for removal. Finally, if any molecules were marked for removal, the method updates the total number of molecules in the system and sets 
        a flag indicating that molecules were removed.
        """

        h = self.headerDict
        N = h["globalnp"]

        #Get volume of cylinder/sphere
        if ends==None:
            ends = [-1e10,1e10]
        elif (rdim != 3):
            V = (ends[1] - ends[0]) * np.pi * radius**2
        else:
            V = (4./3.) * np.pi * radius**3
    
        if (targetdensity > 1e-6):
            molsinV = 0
            #Get density in space
            rmapped = np.zeros(3)
            for n in range(N):
                rmapped[:] = self.r[n,:] - rpos[:]
                #Set zero along direction
                if (rdim != 3):
                    rmapped[rdim] = 0.
                #Spherical or cylindrical radius
                rspherical2 = np.dot(rmapped,rmapped)    #Get position in spherical coordinates
                rspherical = np.sqrt(rspherical2)

                if (rspherical < radius and 
                    self.delmol[n] != 1):
                    #Ends of cylinder
                    if (rdim != 3 and 
                        (self.r[n,rdim] < ends[0] or
                         self.r[n,rdim] > ends[1])):
                        continue

                    molsinV += 1
            density = molsinV / V     
            print("Density in removed region", density, "target=", targetdensity)
        else:
            density = 1.


        deleteratio = targetdensity/density

        #Get density in space
        rmapped = np.zeros(3)
        for n in range(N):
            rmapped[:] = self.r[n,:] - rpos[:]

            #Set zero along direction
            if (rdim != 3):
                rmapped[rdim] = 0.
            #Spherical or cylindrical radius
            rspherical2 = np.dot(rmapped,rmapped)    #Get position in spherical coordinates
            rspherical = np.sqrt(rspherical2)
            #theta = np.acos(rmapped[2]/rspherical)
            #phi = np.atan(rmapped[1]/rmapped[0])

            if (rspherical < radius and
                self.delmol[n] != 1):
                #Ends of cylinder
                #print(rmapped, rmapped[(rdim+1)%3], ends[0], rmapped[(rdim+2)%3], ends[1], 
                if (rdim != 3 and 
                    (self.r[n,rdim] < ends[0] or
                     self.r[n,rdim] > ends[1])):
                    continue

                randno = np.random.random(1)
                if (randno > deleteratio):
                    print(n, self.Nnew, rspherical, radius)
                    self.delmol[n] = 1           
                    self.Nnew -= 1                   
                    self.molecules_deleted = True

    def write_moldata(self, outfile=None, verbose=False):

        #Default to same filename with a 2
        if (outfile is None):
            outfile = self.fname + "2"

        h = self.headerDict
        N = h["globalnp"]

        #Values are the number of values per molecule including all 
        vals = (7 + 3*h["rtrue_flag"] + h["mie_potential"]
                + h["global_numbering"] + 8*h["potential_flag"])
        data = np.zeros(N*vals+ 3*self.Ntethered)

        #Start a new global numbering if any molecules have been deleted
        if (self.molecules_deleted):
            newglob = 1

        #Loop and write all data
        i = 0
        for n in range(N):

            if self.delmol[n] == 1:
                continue

            data[i] = self.tag[n]; i += 1
            data[i:i+3] = self.r[n,:]; i += 3
            data[i:i+3] = self.v[n,:]; i += 3
            #print(n, i, data[i-7:i])

            if (h["rtrue_flag"]):
                data[i:i+3] = self.rtrue[n,:]; i += 3
            if (tag[n] in self.tether_tags):
                data[i:i+3] = self.rtether[n,:]; i += 3
            if (h["mie_potential"]):
                data[i] = self.moltype[n]; i += 1
            if (h["global_numbering"]):
                if (self.molecules_deleted):
                    data[i] = newglob; newglob += 1; i += 1
                else:
                    data[i] = self.globnum[n]; i += 1
            if (h["potential_flag"]):
                data[i:i+8] = self.potdata[n,:]; i += 8

        #Write data to file
        data.tofile(open(outfile, "w+"))

        #If number of molecules has changed, reset to 1x1x1 processors
        if (self.Nnew != h["globalnp"]):
            print("N=", N, "Nnew=", self.Nnew)
            h["globalnp"] = self.Nnew
            h["npx"] = 1; h["npy"] = 1; h["npz"] = 1
            h["procnp0"] = self.Nnew
            proctethernp = 0
            for p in range(self.nproc):
                proctethernp += h["proctethernp"+str(p)]
            h["proctethernp0"] = proctethernp
            delindx = []

        #Update hdata
        for i in range(len(self.hdata)):
            if (verbose or self.hdata[i] != self.headerDict[self.htypes[i]]):
                    print("UPDATE", i, self.htypes[i], "before=", self.hdata[i], 
                          "after=", self.headerDict[self.htypes[i]])
            self.hdata[i] = self.headerDict[self.htypes[i]]
            if self.molecules_deleted:
                if (   ("procnp" in self.htypes[i] and self.htypes[i] != "procnp0")
                    or ("proctethernp" in self.htypes[i] and self.htypes[i] != "proctethernp0")):
                    print("Flagged for Delete", i, self.htypes[i], self.hdata[i]) 
                    delindx.append(i)

        #Delete all other processor tallies if molecules removed
        if self.molecules_deleted:
            for indx in sorted(delindx, reverse=True):
                print("Deleting", self.htypes[indx], self.hdata[indx]) 
                del self.htypes[indx]
                del self.hdata[indx]

        #Update binaryheader
        self.fmtstr = str(self.noints-len(delindx)) + "i"+str(self.ndbls) +"d"+ str(self.ntrlint) + "i"
        binaryheader = struct.pack(self.fmtstr, *self.hdata)

        #Write header at end of file
        #self.size = os.path.getsize(outfile)
        with open(outfile, "ab") as f:
            #f.seek(self.headersize[0])
            f.write(binaryheader)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    fig = plt.figure(); ax = []
    ax.append(fig.add_subplot(1,2,1,projection='3d'))
    ax.append(fig.add_subplot(1,2,2,projection='3d'))

    #Create a final state object
    fs = final_state("./final_state", verbose=True)

    #read the data
    tag, r, v = fs.read_moldata()
    #Plot it
    fs.plot_molecules(ax[0])
    
    # Remove molecules to cut a hole on the film
    # Inputs are position [x,y,z], radius and then direction (x=0, y=1, z=2), ends[bottom,top] of the film top and bottom, and target density
    fs.remove_molecules([0.,0.,0.],8,0,[-6.5,6.5], 0.01)

    #write a new initial_state file
    fs.write_moldata("./final_state_hole", verbose=True)

    #read the new initial_state file and plot
    fst = final_state("./final_state_hole", verbose=True)
    tag, r, v = fst.read_moldata()
    fst.plot_molecules(ax[1])
    plt.show()
