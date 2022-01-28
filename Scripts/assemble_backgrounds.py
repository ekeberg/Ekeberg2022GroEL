import numpy
import h5py
import xfel2146_tools

gain_file = xfel2146_tools.BASE_DIR / "Results/Gain" / "gain_map.h5"
data_dir = xfel2146_tools.BASE_DIR / "Results/Background"
output_dir = xfel2146_tools.BASE_DIR / "Results/AssembledBackground"
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(gain_file, "r") as file_handle:
    mask = numpy.bool8(xfel2146_tools.assemble(file_handle["mask"]))

mask *= numpy.bool8(xfel2146_tools.assemble(xfel2146_tools.mask()))

for runnr in xfel2146_tools.DATA_RUNS:
    this_file = data_dir / f"photon_background_r{runnr:04}.h5"
    output_file = output_dir / f"photon_background_r{runnr:04}.h5"

    print(f"{this_file}: {runnr}", flush=True)

    with h5py.File(this_file, "r") as file_handle:
        photon_background = file_handle["photon_background"][...]
        mask = file_handle["mask"][...]
        npatterns = file_handle["npatterns"][...]
    mask *= xfel2146_tools.mask()
    photon_background = xfel2146_tools.assemble(photon_background)
    mask = numpy.bool8(xfel2146_tools.assemble(mask))
    
    with h5py.File(output_file, "w") as file_handle:
        file_handle.create_dataset("photon_background", data=photon_background)
        file_handle.create_dataset("mask", data=mask)
        file_handle.create_dataset("npatterns", data=npatterns)
