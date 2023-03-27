# Thin-Film-Rupture
Flowmol MD solver was used for the simulations. This is an open source platform developed by one of the authors (E. R. Smith) and can be found here: https://github.com/edwardsmith999/flowmol#flowmol
Once Flowmol is installed: 

1. Run <b>MD_01_equilibration.in</b> with NVT ensemble.
2. Depending on the poking technique:
     (i) Poked-hole: run <b>MD_02_poke_a_hole.in</b> for few time steps (i.e, 1000) with the final_state generated from step 1.
     (ii) Cut-hole: use this python file to cut a hole of target radius and target density to cut a hole. This python code takes the final_state file from the equilibration state and removes molecules according to the user preference:
     # Remove molecules   (currently this is based on a similar setup to the external force)
     # where inputs are position [x,y,z] of the hole center, radius and then direction (x=0, y=1, z=2), ends[bottom,top] of the film, and target density
     i.e.: fs.remove_molecules([0.,0.,0.],8,0,[-6.5,6.5], 0.01)

    #write a new initial_state file
    fs.write_moldata("./final_state_hole", verbose=True)
3. Use the final_state from step 2 as a restart file to run MD_03_production.in with NVE ensemble for sufficiently long time.

# Analysing the data
1. Func_hole_growth.py and Func_surface_functions.py contain the required subroutines called from other .py files.
1. Run hole-growth.py to track the hole radius over time.
2. Run surface-tension.py to measure the local surface tension of the film.
