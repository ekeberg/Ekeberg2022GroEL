import numpy
import os
import re
import h5py
import pathlib
import xfel2146_tools

dark_dir = pathlib.Path(xfel2146_tools.BASE_DIR) / "Results/Dark"
dark_dir.mkdir(parents=True, exist_ok=True)

runnr_all = [144, 149, 152, 158, 168, 169]

for runnr in runnr_all:    
    print(f"{runnr}: ", end="", flush=True)
    data_dark = xfel2146_tools.read_run(runnr, quiet=True)
    print(f"{data_dark.shape[0]}", flush=True)
    dark = data_dark.mean(axis=0)
    with h5py.File(os.path.join(dark_dir, f"dark_r{runnr:04}.h5"), "w") as file_handle:
        file_handle.create_dataset("dark", data=dark)
        file_handle.create_dataset("nimages", data=data_dark.shape[0])
