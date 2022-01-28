#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1023
#SBATCH --partition=fast

import numpy
import sys
import h5py
import os
import pathlib
import scipy.optimize
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools

rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
line = rank

histograms_file = "/scratch/fhgfs/ekeberg/data/xfel2146/histograms.h5"

output_dir = xfel2146_tools.BASE_DIR / "Results/Gain/tmp"
output_dir.mkdir(parents=True, exist_ok=True)

# Fit zero photon peak
    
def fit_function(x, center, sigma, height):
    return height*numpy.exp(-((x - center)**2 / (2*sigma**2)))

parameters_0 = numpy.zeros((1024, 3))

failed_fits = []
for idx in range(1024):

    with h5py.File(histograms_file, "r") as file_handle:
        values = file_handle["histograms_0"][line, idx, :]
        bin_c = file_handle["bin_centers_0"][...]

    try:
        popt, pconv = scipy.optimize.curve_fit(fit_function, bin_c, values, p0=[0, 200, values.max()])
    except RuntimeError:
        failed_fits.append(idx)
        popt = numpy.array([0, 0, 0])

    parameters_0[idx, :] = numpy.array(popt)

    print(f"{idx}: {popt[0]}, {popt[1]}", flush=True)

print(failed_fits, flush=True)


# Fit one photon peak

def fit_function(x, center, sigma, height, base1, base2):
    # return numpy.exp(-base*x) + height*numpy.exp(-((x - center)**2 / (2*sigma**2)))
    return base1/(1+numpy.exp(0.01*(x-base2))) + height*numpy.exp(-((x - center)**2 / (2*sigma**2)))

parameters_1 = numpy.zeros((1024, 5))

failed_fits = []
for idx in range(1024):

    with h5py.File(histograms_file, "r") as file_handle:
        values = file_handle["histograms_1"][line, idx, :]
        bin_c = file_handle["bin_centers_1"][...]

    try:
        popt, pconv = scipy.optimize.curve_fit(fit_function, bin_c, values, p0=[4800, 400, values.max(), values.max(), 4800])
    except RuntimeError:
        failed_fits.append(idx)
        popt = numpy.array([0, 0, 0, 0, 0])

    parameters_1[idx, :] = numpy.array(popt)

    print(f"{idx}: {popt[0]}", flush=True)

print(failed_fits, flush=True)

with h5py.File(output_dir / f"line_{line:04}.h5", "w") as file_handle:
    file_handle.create_dataset("zero_photon", data=parameters_0)
    file_handle.create_dataset("one_photon", data=parameters_1)
