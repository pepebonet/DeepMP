#!/usr/bin/env python3
import os
import h5py
import random
import numpy as np
from collections import OrderedDict

def check_shapes(data1, data2):
    for key in data1.keys():
        if data1[key].shape[1:] != data2[key].shape[1:]:
            raise ValueError("Different shapes for dataset: %s. " % key)


def check_keys(data1, data2):
    if data1.keys() != data2.keys():
        raise ValueError("Files have different datasets.")


def get_size(data):

    sizes = [d.shape[0] for d in data.values()]

    if max(sizes) != min(sizes):
        raise ValueError("Each dataset within a file must have the "
                  "same number of entries!")

    return sizes[0]


def merge_data(data_list):

    data = None

    for f in data_list:
        size = get_size(data_list[f])
        if not data:
            data = data_list[f]
        else:
            check_keys(data, data_list[f])
            check_shapes(data, data_list[f])
            for key in data_list[f]:
                data[key] = np.append(data[key], data_list[f][key], axis=0)

    return data


def load(filename):
    f = h5py.File(filename, 'r')

    data = {}
    for key in f:
        data[key] = f[key][...]
    f.close()
    return data


def save(filename, data):
    f = h5py.File(filename, 'w')
    for key in data:
        f.create_dataset(key, data[key].shape, dtype=data[key].dtype,
                         compression='gzip')[...] = data[key]
    f.close()


def get_set(folder, output, label):
    filelist = [os.path.join(folder, el) for el in os.listdir(folder)]
    random.shuffle(filelist)
    data = OrderedDict()

    for f in filelist:
        data[f] = load(f)

    out_file = os.path.join(output, '{}_combined.h5'.format(label))
    save(out_file, merge_data(data))