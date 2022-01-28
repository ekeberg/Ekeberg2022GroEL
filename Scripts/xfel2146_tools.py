import matplotlib
import numpy
import os
import re
import h5py
import pathlib

# BASE_DIR = ""
try:
    BASE_DIR = pathlib.Path(os.environ["XFEL2146_DIR"])
except KeyError:
    raise EnvironmentError("Set the environment variable XFEL2146_DIR to the directory containing the data")

DATA_DIR = BASE_DIR / "Data"
PNCCD_PATH = "INSTRUMENT/SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output/data/"

DATA_RUNS = [145, 146, 147, 148, 150, 151, 156]
DARK_RUNS = [144, 144, 144, 144, 149, 149, 152]

def read_run(run_nr, start_index=None, max_length=None, quiet=False):
    # Read patterns
    data_dir = os.path.join(DATA_DIR, f"r{run_nr:04}")
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search("PNCCD01", f)]
    file_list.sort()

    if start_index:
        file_list = file_list[((start_index-1) // 500 + 1):]

    patterns = []
    tot_length = 0
    for f in file_list:
        if not quiet:
            print(f, len(file_list))
        with h5py.File(f, "r") as file_handle:
            try:
                data = file_handle[PNCCD_PATH + "image"][...]
            except KeyError:
                if not quiet:
                    print("missing data in", f)
                continue
            patterns.append(data)
            
        tot_length += len(data)
        if max_length and tot_length > max_length:
            break
    if len(patterns) > 1:
        patterns = numpy.vstack(patterns)
    else:
        patterns = patterns[0]

    if start_index:
        patterns = patterns[start_index%500:]
    if max_length:
        patterns = patterns[:max_length]
    return patterns

def file_list(run_nr):
    data_dir = os.path.join(DATA_DIR, "r{:04}".format(run_nr))
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search("PNCCD01", f)]
    # This was used for flat file directory
    # file_list = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if (re.search("PNCCD01", f) and
    #                                                                        re.search("R{:04}".format(run_nr), f))]
    file_list.sort()
    return file_list

def read_file(run_nr, file_nr):
    data_dir = os.path.join(DATA_DIR, "r{:04}".format(run_nr))
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search("PNCCD01", f)]
    file_list.sort()

    if file_nr >= len(file_list):
        raise ValueError(f"file_nr must be < {len(file_list)}")
    f = file_list[file_nr]
    print(f)
    with h5py.File(f, "r") as file_handle:
        patterns = file_handle[PNCCD_PATH + "image"][...]
    return patterns

def assemble(patterns):
    c = [536, 530]
    # c = [530, 536]                                                                                                                                  
    if len(patterns.shape) == 3:
        # panel_1 = patterns[:, c[0]-512:512, :-(c[1]-512-12)]                                                                                        
        panel_1 = patterns[:, c[0]-512:512, 8:]
        panel_2 = patterns[:, 512:-(512+49-c[0]), (c[1]-512):]
        patterns_assembled = numpy.zeros((patterns.shape[0], 1024, 1024))
        patterns_assembled[:, :panel_1.shape[1], :panel_1.shape[2]] = panel_1
        patterns_assembled[:, -panel_2.shape[1]:, :panel_2.shape[2]] = panel_2
        return patterns_assembled
    else:
        panel_1 = patterns[c[0]-512:512, 8:]
        panel_2 = patterns[512:-(512+49-c[0]), (c[1]-512):]
        patterns_assembled = numpy.zeros((1024, 1024))
        patterns_assembled[:panel_1.shape[0], :panel_1.shape[1]] = panel_1
        patterns_assembled[-panel_2.shape[0]:, :panel_2.shape[1]] = panel_2
        return patterns_assembled

def mask():
    mask_file = "/home/ekeberg/Beamtimes/xfel2146/Masks/simple_mask.h5"
    with h5py.File(mask_file, "r") as file_handle:
        mask = numpy.bool8(file_handle["mask"][...])
    return mask

