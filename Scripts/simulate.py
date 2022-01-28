#!/bin/env python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1399
#SBATCH --partition=fast
#SBATCH --gres=gpu:1
#SBATCH --exclude=c002,c023

import numpy
import condor
import os
import h5py
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.environ["XFEL2146_DIR"]) / "Scripts"))
import xfel2146_tools


slurm_rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
slurm_size = int(os.environ["SLURM_ARRAY_TASK_MAX"]) - int(os.environ["SLURM_ARRAY_TASK_MIN"]) + 1


all_files = ["1SS8_H2O_20_cyl.pdb",
             "1SS8_H2O_35_cyl.pdb",
             "1SS8_H2O_45_cyl.pdb",
             "1SS8_H2O_55_cyl.pdb",
             "1SS8_H2O_65_cyl.pdb",
             "1SS8_H2O_hollow_cyl.pdb",
             "1SS8.pdb"]

group_size = slurm_size // len(all_files)
group_rank = slurm_rank % group_size

model_index = slurm_rank // group_size

print(f"{slurm_rank} ({slurm_size}): group: {group_rank} ({group_size}): model: {model_index}", flush=True)

if model_index >= len(all_files):
    print("Slurm array size is not multiple of number of models. This job will be idle.")
    sys.exit()

this_pdb_file = xfel2146_tools.BASE_DIR / "Results/Models" / all_files[model_index]
this_name = this_pdb_file.stem

print(f"{slurm_rank} ({slurm_size}): group: {group_rank} ({group_size}): model: {model_index}", flush=True)

rotations_file = xfel2146_tools.DATA_DIR / "rotations.h5"

output_dir = xfel2146_tools.BASE_DIR / "Results/Templates/tmp"
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(rotations_file, "r") as file_handle:
    all_rotations = file_handle["rotations"][...]
my_rots = all_rotations[group_rank::group_size]
my_indices = numpy.arange(len(all_rotations))[group_rank::group_size]
print(len(my_rots), flush=True)

photon_energy=1200.   # 1.2 keV
intensity=1e-6     # 1 uJ / um^2
detector_distance=0.15 # 20cm or 15cm

h_evs = 4.13566733e-15 # eV s
c = 299792458.0 # m / s
wavelength = h_evs*c/photon_energy


downsample = 8
source = condor.Source(wavelength=wavelength, # m
                       pulse_energy=intensity, # J
                       focus_diameter=1e-6) # m
detector = condor.Detector(distance=detector_distance,
                           pixel_size=downsample*75e-6,
                           nx=1024//downsample, ny=1024//downsample, # Detector size in pixels
                           cx=512//downsample-0.5, cy=512//downsample-0.5) # Detector center in pixels

for index, rot in zip(my_indices, my_rots):
    print(f"{index}: {rot}", flush=True)
    particle = condor.ParticleAtoms(pdb_filename=str(this_pdb_file), rotation_values=rot, rotation_formalism="quaternion")
    experiment = condor.Experiment(source, {"particle_atoms": particle}, detector)
    result = experiment.propagate()

    # pattern = result["entry_1"]["data_1"]["data"].transpose()
    # fourier = result["entry_1"]["data_1"]["data_fourier"].transpose()
    pattern = result["entry_1"]["data_1"]["data"]
    fourier = result["entry_1"]["data_1"]["data_fourier"]

    with h5py.File(output_dir / f"pattern_{this_name}_{index:06}.h5", "w") as file_handle:
        file_handle.create_dataset("pattern", data=pattern)
        file_handle.create_dataset("rotation", data=rot)

print("Done", flush=True)
