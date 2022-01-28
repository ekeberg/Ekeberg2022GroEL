import numpy
import h5py
import xfel2146_tools

data_file = xfel2146_tools.BASE_DIR / "Results/Fit/results.h5"


models = ["1SS8_H2O_20_cyl",
          "1SS8_H2O_35_cyl",
          "1SS8_H2O_45_cyl",
          "1SS8_H2O_55_cyl",
          "1SS8_H2O_65_cyl",
          "1SS8_H2O_hollow_cyl",
          "1SS8"]

signal = []
background = []
fit = []
shift_x = []
shift_y = []


with h5py.File(data_file, "r") as file_handle:
    for this_model in models:
        signal.append(float(file_handle[this_model + "/signal"][()]))
        background.append(float(file_handle[this_model + "/background"][()]))
        fit.append(float(file_handle[this_model + "/fit"][()]))
        shift_x.append(int(file_handle[this_model + "/shift_x"][()]))
        shift_y.append(int(file_handle[this_model + "/shift_y"][()]))

print("\n".join([str(e) for e in zip(fit, shift_x, shift_y, signal, background)]))
