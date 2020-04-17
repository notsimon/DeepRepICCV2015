"""
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
"""
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import scipy
from scipy.io import matlab
import h5py

import cortex.utils


# ---------------------------------------------------------------------------#


def load_single_set(cFrames, set_num):
    datset = []
    for i in range(0, 20):
        xx = cFrames[0, 20 * set_num + i]
        datset.append(xx)
    npdataset = np.array(datset, ndmin=4)
    return npdataset


# ---------------------------------------------------------------------------#


def load_rep_dataset(filename):
    mat = matlab.loadmat(filename)
    cFrames = mat["all_cFrames"]
    labels = mat["labels"]
    n_sets = labels.shape[1]
    data_x = load_single_set(cFrames, 0)
    for i in range(1, labels.shape[1]):
        data_x = np.append(data_x, load_single_set(cFrames, i), axis=0)
    data_x = (data_x * 255.0).astype(np.uint8)
    labels = labels.flatten().astype(np.int)
    return data_x, labels


# ---------------------------------------------------------------------------#

in_dir = "/home/tomrunia/dev/python/DeepRepICCV2015/out/mat/"
out_dir = "/home/tomrunia/dev/python/DeepRepICCV2015/out/h5/"

mat_files = cortex.utils.find_files(in_dir, "mat")

for mat_file in mat_files:

    print("Processing file: {}".format(mat_file))
    frames, labels = load_rep_dataset(mat_file)
    filename = cortex.utils.basename(mat_file)
    out_file = os.path.join(out_dir, filename + ".h5")

    with h5py.File(out_file, "w") as hf:
        dset_videos = hf.create_dataset("videos", data=frames)
        dset_labels = hf.create_dataset("labels", data=labels)

print("Done.")
