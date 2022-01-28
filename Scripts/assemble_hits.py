import numpy
import h5py
import xfel2146_tools

gain_file = xfel2146_tools.BASE_DIR / "Results/Gain" / "gain_map.h5"
data_dir = xfel2146_tools.BASE_DIR / "Results/Hits"
output_dir = xfel2146_tools.BASE_DIR / "Results/AssembledHits"
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(gain_file, "r") as file_handle:
    mask = numpy.bool8(xfel2146_tools.assemble(file_handle["mask"]))

mask *= numpy.bool8(xfel2146_tools.assemble(xfel2146_tools.mask()))

for runnr in xfel2146_tools.DATA_RUNS:
    this_file = data_dir / f"hits_r{runnr:04}.h5"
    output_file = output_dir / f"hits_r{runnr:04}.h5"

    print(f"{this_file}: {runnr}", flush=True)

    with h5py.File(this_file, "r") as file_handle:
        patterns_ph = file_handle["patterns_ph"][...]
        patterns_cm = file_handle["patterns_cm"][...]
        patterns = file_handle["patterns"][...]
    assembled_ph = xfel2146_tools.assemble(patterns_ph)
    assembled_cm = xfel2146_tools.assemble(patterns_cm)
    assembled = xfel2146_tools.assemble(patterns)
    
    with h5py.File(output_file, "w") as file_handle:
        file_handle.create_dataset("patterns_ph", data=assembled_ph)
        file_handle.create_dataset("patterns_cm", data=assembled_cm)
        file_handle.create_dataset("patterns", data=assembled)
        file_handle.create_dataset("mask", data=mask)
