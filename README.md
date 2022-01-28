This repository contains the code required for analysis of Ekeberg. T
et al. Observation of a single protein by ultrafast X-ray
diffraction. (2022).  It requires you to download the corresponding
data and place it in a directory called Data in this directory. You
should also create an environment variable called XFEL2146_DIR that
points to this directory.

Bellow follows a short description of each of the scripts

# Dark files
Run Scripts/prepare_dark.py
Calculates the detector baseline

# Gain fitting
Run Scripts/fit_gain_parameters.py using sbatch
Fit the zero and one photon peaks to detector readout histograms to get the detector response.

# Gain map
Run Scripts/make_gain_map.py
Combine the fitting results into a single file

# Hitfinding 1
Run Scripts/count_lit_pixels.py using sbatch
Counts the number of lit pixels in each frame. This is the first step of hitfinding

# Hitfinding 2
Run Scripts/export_hits.py using sbatch
Save all patterns with lit pixels above a threshold

# Assemble hits
Run Scripts/assemble_hits.py
Combine the detector modules for hits

# Background
Run Scripts/photon_background.py using sbatch
Sum up data from runs without sample to get an average photon background

# Assemble background
Run Scripts/assemble_background.py
Combine detector modules for the background

# Generate models
Run Scripts/generate_water_models.py
Combine GroEL and water to create the density models

# Simulate
Run Scripts/simulate.py using sbatch
Simulate diffraction data from pure GroEL and from the different density models

# Combine simulations
Run Scripts/combine_simulations.py
Combine the resuls from the simulation into easier-to-handle files

# Template matching
Run Scripts/fit.py using sbatch
Find the best fitting orientation and translation for each structure model

# Combine results from template matching
Run Scripts/combine_fits.py
Combine the results from template matching into a single file

