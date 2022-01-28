import numpy
import h5py
import os
import re
import xfel2146_tools

input_dir = xfel2146_tools.BASE_DIR / "Results/Templates/tmp"
output_dir = xfel2146_tools.BASE_DIR / "Results/Templates"

# all_names = ["1SS8_H2O_20_cyl",
#              "1SS8_H2O_35_cyl",
#              "1SS8_H2O_45_cyl",
#              "1SS8_H2O_55_cyl",
#              "1SS8_H2O_65_cyl",
#              "1SS8_H2O_hollow_cyl",
#              "1SS8"]
all_names = ["1SS8_H2O_35_cyl",
             "1SS8_H2O_45_cyl",
             "1SS8_H2O_55_cyl",
             "1SS8_H2O_65_cyl",
             "1SS8_H2O_hollow_cyl",
             "1SS8"]

number_of_rotations = 25680

patterns = numpy.zeros((number_of_rotations, 128, 128))
rotations = numpy.zeros((number_of_rotations, 4))

for this_name in all_names:
    print()
    print()
    print("   ***   " + this_name + "   ***   ")
    print()
    print()

    
    files = [input_dir / f"pattern_{this_name}_{index:06}.h5" for index in range(number_of_rotations)]

    for index, this_file in enumerate(files):
        if index % 100 == 0:
            print(f"{this_name}: {this_file}")
        with h5py.File(this_file, "r") as file_handle:
            patterns[index, :, :] = file_handle["pattern"][...]
            rotations[index, :] = file_handle["rotation"][...]

    with h5py.File(output_dir / f"patterns_{this_name}.h5", "w") as file_handle:
        file_handle.create_dataset("patterns", data=patterns)
        file_handle.create_dataset("rotations", data=rotations)
