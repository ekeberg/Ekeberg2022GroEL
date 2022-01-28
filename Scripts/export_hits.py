#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --array=0-6
#SBATCH --partition=fast
#SBATCH --mem=20G

import numpy
import os
import h5py
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools

rank = int(os.environ["SLURM_ARRAY_TASK_ID"])

runnr = xfel2146_tools.DATA_RUNS[rank]
runnr_dark = xfel2146_tools.DARK_RUNS[rank]
print(f"runnr = {runnr}", flush=True)
print(f"runnr_dark = {runnr_dark}", flush=True)

dark_file = xfel2146_tools.BASE_DIR / "Results/Dark" / f"dark_r{runnr_dark:04}.h5"
gain_file = xfel2146_tools.BASE_DIR / "Results/Gain" / "gain_map.h5"
lit_pixels_file = xfel2146_tools.BASE_DIR / "Results/LitPixels" / f"lit_pixels_r{runnr:04}.h5"
output_dir = xfel2146_tools.BASE_DIR / "Results/Hits"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"hits_r{runnr:04}.h5"


with h5py.File(dark_file, "r") as file_handle:
    dark = file_handle["dark"][...]

with h5py.File(gain_file, "r") as file_handle:
    peak_0_pos = file_handle["peak_0_pos"][...]
    peak_1_pos = file_handle["peak_1_pos"][...]
    mask = file_handle["mask"][...]

mask *= xfel2146_tools.mask()
    
def h5py_append(dataset, data):
    # print("append ", data.shape, " to ", dataset, flush=True)
    old_size = dataset.shape[0]
    dataset.resize((old_size+data.shape[0], ) + dataset.shape[1:])
    dataset[old_size:] = data[:]

with h5py.File(lit_pixels_file, "r") as file_handle:
    lit_pixels = file_handle["lit_pixels"][...]

lit_pixel_threshold = numpy.percentile(lit_pixels, 99)

hits = lit_pixels > lit_pixel_threshold
file_list = xfel2146_tools.file_list(runnr)

with h5py.File(output_file, "w") as file_handle:
    file_handle.create_dataset("lit_pixels", data=lit_pixels[hits])
    file_handle.create_dataset("patterns_raw", (0, 1024, 1024), maxshape=(None, 1024, 1024), dtype="f8", chunks=True)
    file_handle.create_dataset("patterns", (0, 1024, 1024), maxshape=(None, 1024, 1024), dtype="f8", chunks=True)
    file_handle.create_dataset("patterns_cm", (0, 1024, 1024), maxshape=(None, 1024, 1024), dtype="f8", chunks=True)
    file_handle.create_dataset("patterns_ph", (0, 1024, 1024), maxshape=(None, 1024, 1024), dtype="f8", chunks=True)


counter = 0    
for f in file_list:
# for filenr in range(2):
    print(f, flush=True)
    with h5py.File(f, "r") as file_handle:
        file_size = file_handle[xfel2146_tools.PNCCD_PATH + "image"].shape[0]
        my_hits = hits[counter:counter+file_size]
        counter += file_size
        if my_hits.sum() == 0:
            continue
        print(f"{my_hits.sum()} hits of {len(my_hits)} ({counter} -> {counter+file_size})", flush=True)
        patterns_raw = file_handle[xfel2146_tools.PNCCD_PATH + "image"][my_hits, :, :]
        

    patterns = patterns_raw - dark[numpy.newaxis, :, :]

    # Common mode correction
    patterns_reshape = patterns.reshape((patterns.shape[0], 2, 512, 1024))
    patterns_cm = (patterns_reshape -
                   numpy.median(patterns_reshape, axis=2)[:, :, numpy.newaxis, :]).reshape((patterns.shape[0], 1024, 1024))

    #Photon conversion
    patterns_ph = patterns_cm.copy()
    patterns_ph -= peak_0_pos[numpy.newaxis, ...]
    patterns_ph /= peak_1_pos[numpy.newaxis, ...]
    patterns_ph[:, ~mask] = 0
    patterns_ph[patterns_ph < 0.75] = 0
    numpy.round(patterns_ph, out=patterns_ph)

    patterns_ph *= mask[numpy.newaxis, ...]
    
    with h5py.File(output_file, "r+") as file_handle:
        h5py_append(file_handle["patterns_raw"], patterns_raw)
        h5py_append(file_handle["patterns"], patterns)
        h5py_append(file_handle["patterns_cm"], patterns_cm)
        h5py_append(file_handle["patterns_ph"], patterns_ph)
