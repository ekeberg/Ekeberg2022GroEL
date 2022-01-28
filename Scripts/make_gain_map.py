import numpy
import h5py
import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools

data_dir = xfel2146_tools.BASE_DIR / "Results/Gain/tmp"
output_file = xfel2146_tools.BASE_DIR / "Results/Gain/gain_map.h5"

peak_1_pos = numpy.zeros((1024, 1024))
peak_1_sigma = numpy.zeros((1024, 1024))
peak_0_pos = numpy.zeros((1024, 1024))
peak_0_sigma = numpy.zeros((1024, 1024))
            
for line in range(1024):
    print(line)
    #try:
    with h5py.File(data_dir / f"line_{line:04}.h5", "r") as file_handle:
        peak_0_pos[line, :] = file_handle["zero_photon"][:, 0]
        peak_0_sigma[line, :] = file_handle["zero_photon"][:, 1]
        peak_1_pos[line, :] = file_handle["one_photon"][:, 0]
        peak_1_sigma[line, :] = file_handle["one_photon"][:, 1]
            
    # except OSError:
    #     print(f"Warning, no data for line {line}")

mask = ((peak_1_pos - peak_0_pos) < 5500) * ((peak_1_pos - peak_0_pos) > 4000) * (abs(peak_0_sigma) < 500) * (abs(peak_1_sigma) < 1000)


with h5py.File(output_file, "w") as file_handle:
    file_handle.create_dataset("peak_0_pos", data=peak_0_pos)
    file_handle.create_dataset("peak_0_sigma", data=peak_0_sigma)
    file_handle.create_dataset("peak_1_pos", data=peak_1_pos)
    file_handle.create_dataset("peak_1_sigma", data=peak_1_sigma)
    file_handle.create_dataset("mask", data=mask)
