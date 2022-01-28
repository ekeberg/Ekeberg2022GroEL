#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-6
#SBATCH --partition=fast
#SBATCH --mem=30G
#SBATCH --exclude=c002

import numpy
import os
import h5py
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools

rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
index = rank

runnr = xfel2146_tools.DATA_RUNS[index]
runnr_dark = xfel2146_tools.DARK_RUNS[index]
print(f"runnr = {runnr}", flush=True)
print(f"runnr_dark = {runnr_dark}", flush=True)

dark_file = xfel2146_tools.BASE_DIR / "Results/Dark" / f"dark_r{runnr_dark:04}.h5"
gain_file = xfel2146_tools.BASE_DIR / "Results/Gain" / "gain_map.h5"
lit_pixels_file = xfel2146_tools.BASE_DIR / "Results/LitPixels" / f"lit_pixels_r{runnr:04}.h5"
output_dir = xfel2146_tools.BASE_DIR / "Results/Background"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"photon_background_r{runnr:04}.h5"
# output_file = output_dir / f"photon_background_transpose_r{runnr:04}.h5"

miss_threshold = 10

with h5py.File(dark_file, "r") as file_handle:
    dark = file_handle["dark"][...]

with h5py.File(gain_file, "r") as file_handle:
    mask = file_handle["mask"][...]
    peak_0_pos = file_handle["peak_0_pos"][...]
    peak_1_pos = file_handle["peak_1_pos"][...]
    peak_0_sigma = file_handle["peak_0_sigma"][...]
    peak_1_sigma = file_handle["peak_1_sigma"][...]
    
npatterns = 0
photon_background = numpy.zeros(dark.shape)
counter = 0

with h5py.File(lit_pixels_file, "r") as file_handle:
    lit_pixels = file_handle["lit_pixels"][...]

nfiles = len(xfel2146_tools.file_list(runnr))
for filenr in range(nfiles):
    # for filenr in range(2):
    # print(filenr, end="", flush=True)
    # print(filenr, flush=True)
    print(f"{filenr} ({nfiles})", flush=True)
    patterns_raw = xfel2146_tools.read_file(runnr, filenr)
    npatterns_raw = len(patterns_raw)
    my_misses = lit_pixels[counter:counter+npatterns_raw] < miss_threshold
    counter += npatterns_raw
    if my_misses.sum() == 0:
        continue
    patterns_raw = patterns_raw[my_misses]
    
    patterns_raw = patterns_raw
    patterns = patterns_raw - dark[numpy.newaxis, :, :]
    print(f" {len(patterns)}", flush=True)

    # Common mode correction
    print("Common mode correction", flush=True)
    patterns_reshape = patterns.reshape((patterns.shape[0], 2, 512, 1024))
    patterns_cm = (patterns_reshape -
                   numpy.median(patterns_reshape, axis=2)[:, :, numpy.newaxis, :]).reshape((patterns.shape[0], 1024, 1024))

    #Photon conversion
    print("Photon conversion", flush=True)
    patterns_ph = (patterns_cm - peak_0_pos) / (peak_1_pos - peak_0_pos)
    patterns_ph[patterns_ph < 0.75] = 0
    patterns_ph = numpy.round(patterns_ph)

    patterns_ph[..., ~mask] = 0

    photon_background += patterns_ph.sum(axis=0)
    npatterns += len(patterns_ph)

photon_background /= npatterns
    
with h5py.File(output_file, "w") as file_handle:
    file_handle.create_dataset("photon_background", data=photon_background)
    file_handle.create_dataset("mask", data=mask)
    file_handle.create_dataset("npatterns", data=npatterns)
