#!/usr/bin/env python3
import os
import ast
import h5py
import click
import numpy as np


def get_fast5(fastq_info):
    dict_read = ast.literal_eval(fastq_info)
    return dict_read['filename']


def get_fast5_structure(fast5, fastq_info):
    basecall_path = 'Analyses/Basecall_1D_000/BaseCalled_template/'
    with h5py.File(fast5, 'a') as data:
        if basecall_path not in data:
            data.create_group(basecall_path)

        if os.path.join(basecall_path, 'Fastq') not in data:
            dt = h5py.string_dtype('ascii')
            dset = data.create_dataset(
                os.path.join(basecall_path, 'Fastq'), (), dtype=dt
            )
            dset[()] = '\n'.join(fastq_info)


def load_txt(path):
    items = []
    with open(path, 'r') as f:
        for line in f:
            items.append(line.strip())
        return items


@click.command(short_help='Script to add fastqs to the fast5s')
@click.option(
    '-f5', '--fast5s_path', help='path to fast5 reads'
)
@click.option(
    '-fq', 'fastqs_path', help='path to fastqs basecalled by flappie'
)
def main(fast5s_path, fastqs_path):
    fastqs = load_txt(fastqs_path)
    for i in range(0, len(fastqs), 4):
        fastq = fastqs[i: i + 4]
        fast5 = get_fast5(fastq[0].split('  ', 1)[-1])
        fast5_file = os.path.join(fast5s_path, fast5)
        if os.path.isfile(fast5_file):
            get_fast5_structure(fast5_file, fastq)


if __name__ == "__main__":
    main()
