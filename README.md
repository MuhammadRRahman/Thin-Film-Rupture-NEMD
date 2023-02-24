# Thin-Film-Rupture
Flowmol MD solver was used for the simulations. This is an open source platform developed by one of the authors (E. R. Smith) and can be found here: https://github.com/edwardsmith999/flowmol#flowmol
Once Flowmol is installed: 

1. Run <b>MD_01_equilibration.in</b> with NVT ensemble.
2. Depending on the poking technique:
     (i) Poked-hole: run <b>MD_02_poke_a_hole.in</b> for few time steps (i.e, 1000) with the final_state generated from step (1).
3. Use the final_state from step 2 as a restart file to run MD_03_production.in with NVE ensemble for sufficiently long time.

# Analysing the data
1. Func_hole_growth.py and Func_surface_functions.py contains the required subroutines called from other .py files.
1. Run hole-growth.py to track the hole radius over time.
2. Run surface-tension.py to measure the local surface tension of the film.
