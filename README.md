Codes to reproduce data for: 

# Life and Death of a Thin Liquid Film
   by
   Muhammad Rizwanur Rahman, Li Shen, James P. Ewen, David M. Heyes, Daniele Dini, and, Edward R. Smith (under review).
   Initial preprint: arXiv preprint: https://doi.org/10.48550/arXiv.2311.00419, data-files: https://doi.org/10.5281/zenodo.12633918
   
# Non-Equilibrium Molecular Simulations of Thin Film Rupture
   by 
   Muhammad Rizwanur Rahman, Li Shen, James P. Ewen, Benjamin Collard, David M. Heyes, Daniele Dini, and, Edward R. Smith.
   Journal of Chemical Physics (Communication), April 2023.
   https://aip.scitation.org/doi/10.1063/5.0149974

** This repository will be further developed. 


# MD Solver
Flowmol MD solver was used for the simulations. This is an open source platform developed by one of the authors (E. R. Smith) and can be found here: https://github.com/edwardsmith999/flowmol#flowmol
Once Flowmol is installed: 

1. Run <b>MD_01_equilibration.in</b> with NVT ensemble.
2. Depending on the poking technique:

     (i) <b>Poked-hole:</b> run <b>MD_02_poke_a_hole.in</b> for few time steps (i.e, 1000) with the final_state generated from step 1.
     
     (ii) <b>Cut-hole:</b> use <b>MD_02_cut_a_hole.py</b> to cut a hole of target radius and target density. This python code takes the final_state file from the equilibration state and removes molecules according to the user preference:
     
     Remove molecules (currently this is based on a similar setup to the external force)
     where inputs are position [x,y,z] of the hole center, radius and then direction (x=0, y=1, z=2), ends[bottom,top] of the film, and target density
     i.e.: fs.remove_molecules([0.,0.,0.],8,0,[-6.5,6.5], 0.01). The python code writes a new initial_state file to be used in the production phase.
     
     NB. MD_02_cut_a_hole.py requires the final_state (from the equilibration phase) file in the same directory. 

3. Use the final_state from step 2 as a restart file to run MD_03_production.in with NVE ensemble for sufficiently long time.
4. For spontaneous hole formation, skip step 2, and start production phase without externally poking or cutting a hole in the film. To attain rupture within computationally feasible time, start with thinner films.
5. To start the production phase, an NVT equilibrated state file is used. Some of these state files for spontaneous rupture cases, as well as for synthetically damaged hole growth cases are made publicly available at zenodo, please follow this link: https://doi.org/10.5281/zenodo.12633918
6. To save finalstates frequently for the investigation of the memory window, set the RESCUE_SNAPSHOT_FREQ to a lower value. 

# Analysing the data

1. Func_hole_growth.py and Func_surface_functions.py contain the required subroutines called from other .py files.
1. hole-growth.py  tracks the hole radius over time.
2. surface-tension.py  measures the local surface tension of the film.
3. surface-tension-mapped-pressure-tensor.py measures local surface tension with pressuree-tensor rotated along the rim profile. Also tracks the evolution      of the film profile with time.
4. radial-averaged-density.py creates images of the radial averaged density field setting the origin at the hole center (0,0,0).
5. Temperature-Velocity-radial-averaged.py plots the radial averaged velocity and temperature field.
   
** These script files will be further developed soon for enhanced readability, and for easy-to-follow work flow.
