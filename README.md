# Thin-Film-Rupture
Flowmol MD solver was used for the simulations. This is an open source platform developed by one of the authors (E. R. Smith) and can be found here: https://github.com/edwardsmith999/flowmol#flowmol

1. Initially, run <b>film.in</b> with NVT ensemble.
2. Depending on the poking technique:
     (i) Poked-hole: run <b>poked-film.in</b> for few time steps (i.e, 1000)
    (ii) Cut-hole: run <b>cut-film.py</b> 
3. Use the final_state from step 2 as a restart file to run production.in in NVE ensemble

# Analysing the data
1. Run hole-growth.py to track the hole radius over time
2. Run surface-tension.py to measure the bulk surface tension of the film
3. Run local-surface-tension.py to measure local surface tension
4. Run rotated-surface-tension.py to measure surface tension with pressure tensor rotated along the local normal and tangential directions.
5. Rup plot.py to regenrate the plots of teh paper
