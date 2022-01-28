#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0,1
#SBATCH --partition=regular

# Array size was nmodels * npatterns.
# Now only one pattern. Could potentially separate different each fit over multiple.
# Or just go with one per model.

import numpy
import h5py
import pickle
import os
import pathlib

import sys
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import template_matching
import xfel2146_tools

runnr = 151
pattern_index = 281

shift_radius = 6

slurm_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

models = ["1SS8",
          "1SS8_H2O_20_cyl",
          "1SS8_H2O_35_cyl",
          "1SS8_H2O_45_cyl",
          "1SS8_H2O_55_cyl",
          "1SS8_H2O_65_cyl",
          "1SS8_H2O_hollow_cyl"]
model = models[slurm_index]

output_dir = xfel2146_tools.BASE_DIR / "Results/Fit/tmp"
output_dir.mkdir(parents=True, exist_ok=True)
background_file = xfel2146_tools.BASE_DIR / "Results/Background" / f"photon_background_r{runnr:04}.h5"
pattern_file = xfel2146_tools.BASE_DIR / "Results/AssembledHits" / f"hits_r{runnr:04}.h5"
templates_file = xfel2146_tools.BASE_DIR / "Results/Templates" / f"patterns_{model}.h5"

print(slurm_index, model, flush=True)

template_density_n = 8

with h5py.File(background_file, "r") as file_handle:
    background_large = file_handle["photon_background"][...]
background = background_large.reshape((128, 8, 128, 8)).sum(axis=(1, 3))

with h5py.File(pattern_file, "r") as file_handle:
    pattern_large = file_handle["patterns_ph"][pattern_index]
    mask_large = file_handle["mask"][...]

pattern = pattern_large.reshape((128, 8, 128, 8)).sum(axis=(1, 3))
mask = mask_large.reshape((128, 8, 128, 8)).sum(axis=(1, 3)) == 64
x, y = numpy.meshgrid(numpy.arange(128)-63.5, numpy.arange(128)-63.5, indexing="ij")
mask *= numpy.sqrt(x**2 + y**2) > 5
mask *= numpy.sqrt(x**2 + y**2) < 40

with h5py.File(templates_file, "r") as file_handle:
    templates = file_handle["patterns"][...]
    template_rotations = file_handle["rotations"][...]


matcher = template_matching.MatcherBGCenter(templates, mask, background, range(-shift_radius, shift_radius+1), quiet=False)
fit_param = matcher.match(pattern)

print(model, pattern_index, fit_param["fit"], fit_param["signal"], fit_param["background"], fit_param["shift_x"], fit_param["shift_y"], flush=True)

fit_param["slurm_index"] = slurm_index
fit_param["model"] = model

with open(output_dir /  f"{slurm_index:04}.p", "wb") as file_handle:
    pickle.dump(fit_param, file_handle)
