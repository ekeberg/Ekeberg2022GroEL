import numpy
import pickle
import os
import h5py
import xfel2146_tools

n = 8

input_dir = xfel2146_tools.BASE_DIR / "Results/Fit/tmp"
files = os.listdir(input_dir)
files.sort()
output_file = xfel2146_tools.BASE_DIR / "Results/Fit" / "results.h5"


d = [pickle.load(open(input_dir / this_file, "rb")) for this_file in files]

shift_x = numpy.array([d0["shift_x"] for d0 in d])
shift_y = numpy.array([d0["shift_y"] for d0 in d])
fit = numpy.array([d0["fit"] for d0 in d])
signal = numpy.array([d0["signal"] for d0 in d])
background = numpy.array([d0["background"] for d0 in d])
index = numpy.array([d0["index"] for d0 in d])
slurm_index = numpy.array([d0["slurm_index"] for d0 in d])
model = numpy.array([d0["model"] for d0 in d])
# pattern_index = numpy.array([d0["pattern_index"] for d0 in d])
# pattern_index_in_run = numpy.array([d0["pattern_index_in_run"] for d0 in d])


models = numpy.unique(model)

with h5py.File(output_file, "w") as file_handle:
    for this_model in models:
        selection = model == this_model
        model_group = file_handle.create_group(this_model)
        model_group.create_dataset("shift_x", data=shift_x[selection])
        model_group.create_dataset("shift_y", data=shift_y[selection])
        model_group.create_dataset("fit", data=fit[selection])
        model_group.create_dataset("signal", data=signal[selection])
        model_group.create_dataset("background", data=background[selection])
        model_group.create_dataset("index", data=index[selection])
        model_group.create_dataset("slurm_index", data=slurm_index[selection])

