# Welcome to the SHAMPOO GitHub repository
SHAMPOO (**S**toc**HA**stic **M**onomer **P**r**O**cess**O**r) is a model simulating the evolution of individual dust 
particles in a stochastic fashion, faciliating a Monte-Carlo-like approach towards dust evolution in protoplanetary disks.

Developed by Mark Oosterloo (University of Groningen (UG), Vrije Universiteit Amsterdam (VUA)) in collaboration with 
Inga Kamp (UG) and Wim van Westerenen (VUA)

# Introduction
SHAMPOO provides a framework to explore the effects of dynamical, collisional and ice evolution processes combined on an 
individual dust particle in a planet-forming disk. SHAMPOO treats these processes in a fully coupled fashion. For the 
surrounding disk environment, the code relies on the output of [ProDiMo](https://prodimo.iwf.oeaw.ac.at/) disk models, 
and assumes the disk remains both physically and chemically static over the duration of the simulation. This makes SHAMPOO 
suitable for predicting properties of dust particles in planet-forming region on timescales shorter than 100 000 yr. In 
addition to the SHAMPOO code itself, this repository contains the tools to simulate and analyze the trajectories of 
monomers in similar fashion as in [Oosterloo et al. 2023](https://ui.adsabs.harvard.edu/abs/2023DPS....5550002O/abstract).

# Getting started
In order to run SHAMPOO, simply create a venv from the `requirements.txt` file. SHAMPOO can be executed in a notebook 
fashion. SHAMPOO uses Christian Rab's [ProDiMoPy](https://gitlab.astro.rug.nl/prodimo/prodimopy) to read the output of
ProDiMo disk models.

Some example scripts are included in the ```scripts``` folder. 
- `disk_reader.py` - Uses ProDiMoPy to read the output of a ProDiMo simulation and store it in a pickle `.pkl` file.
- `run_shampoo.py` - Runs SHAMPOO using the disk model stored in the `.pkl` file created with `disk_reader.py`.
- `shampooIndividualMonomers.ipynb` - Can do everything the above scripts can, and in addition also visualize the results
for an individual dust monomer.