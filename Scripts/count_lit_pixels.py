#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-6
#SBATCH --partition=fast
#SBATCH --mem=30G

import numpy
import os
import re
import h5py
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools

rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
runnr = xfel2146_tools.DATA_RUNS[rank]
runnr_dark = xfel2146_tools.DARK_RUNS[rank]

dark_file = xfel2146_tools.BASE_DIR / "Results/Dark" / f"dark_r{runnr_dark:04}.h5"
gain_file = xfel2146_tools.BASE_DIR / "Results/Gain" / "gain_map.h5"
output_dir = xfel2146_tools.BASE_DIR / "Results/LitPixels"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"lit_pixels_r{runnr:04}.h5"

print("Read dark", flush=True)
with h5py.File(dark_file, "r") as file_handle:
    dark = file_handle["dark"][...]

print("Read gain", flush=True)
with h5py.File(gain_file, "r") as file_handle:
    peak_0_pos = file_handle["peak_0_pos"][...]
    peak_1_pos = file_handle["peak_1_pos"][...]
    mask = file_handle["mask"][...]

print("Read mask", flush=True)
mask *= xfel2146_tools.mask()

print("Read everything and start loop", flush=True)

lit_pixels = []

file_list = xfel2146_tools.file_list(runnr)
# file_list = file_list[:2]

for filenr in range(len(file_list)):
# for filenr in range(2):
    # print(filenr, end="", flush=True)
    print(filenr, flush=True)
    patterns_raw = xfel2146_tools.read_file(runnr, filenr)
    patterns = patterns_raw - dark[numpy.newaxis, :, :]
    print(f" {len(patterns)}")

    # Common mode correction
    print("Common mode correction")
    patterns_reshape = patterns.reshape((patterns.shape[0], 2, 512, 1024))
    patterns_cm = (patterns_reshape -
                   numpy.median(patterns_reshape, axis=2)[:, :, numpy.newaxis, :]).reshape((patterns.shape[0], 1024, 1024))

    #Photon conversion
    # print("Photon conversion")
    # patterns_ph = patterns_cm / stuff.adu_per_photon
    # patterns_ph[patterns_ph < 0.75] = 0
    # patterns_ph = numpy.round(patterns_ph)
    
    #Photon conversion
    print("Photon conversion")
    # patterns_ph = (patterns_cm - peak_0_pos[numpy.newaxis, ...]) / peak_1_pos[numpy.newaxis, ...]
    patterns_ph = patterns_cm.copy()
    patterns_ph -= peak_0_pos[numpy.newaxis, ...]
    patterns_ph /= peak_1_pos[numpy.newaxis, ...]
    print("Photon conversion 2")
    patterns_ph[:, ~mask] = 0
    print("Photon conversion 3")
    patterns_ph[patterns_ph < 0.75] = 0
    print("Photon conversion 4")
    # patterns_ph = numpy.round(patterns_ph)
    numpy.round(patterns_ph, out=patterns_ph)
    
    print("Count lit pixels")
    lit_pixels.append((patterns_ph > 2).sum(axis=(1, 2)))

lit_pixels = numpy.hstack(lit_pixels)

print("Write to file")
with h5py.File(output_file, "w") as file_handle:
    file_handle.create_dataset("lit_pixels", data=lit_pixels)
print("Done")

