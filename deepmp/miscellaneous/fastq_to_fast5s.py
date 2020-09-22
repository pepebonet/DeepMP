#!/usr/bin/env python3
import os
import ast
import h5py
import click
import functools
import numpy as np
from multiprocessing import Pool

from tqdm import tqdm

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
            import pdb;pdb.set_trace()
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


def fastq_to_fast5(i, fastqs, fast5s_path):
    fastq = fastqs[i: i + 4]
    fast5 = get_fast5(fastq[0].split('  ', 1)[-1])
    
    for el in [x[0] for x in os.walk(fast5s_path)][1:]: 
        fast5_file = os.path.join(el, fast5)
        
        if os.path.isfile(fast5_file):
            get_fast5_structure(fast5_file, fastq)
            break


@click.command(short_help='Script to add fastqs to the fast5s')
@click.option(
    '-f5', '--fast5s_path', help='path to fast5 reads'
)
@click.option(
    '-ft', '--fastq_type', required=True,
    type=click.Choice(['folder', 'file']),
    help='is a fastq file or a folder with several fastqs?'
)
@click.option(
    '-fq', '--fastqs_path', help='path to fastqs basecalled by flappie'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of cores to be used'
)
def main(fastq_type, fast5s_path, fastqs_path, cpus):

    if fastq_type == 'file':
        fastqs = load_txt(fastqs_path)
        for i in tqdm(range(0, len(fastqs), 4)):
            fastq_to_fast5(i, fastqs, fast5s_path)
        
    else:
        for el in tqdm(os.listdir(fastqs_path)):

            if os.path.isfile(os.path.join(fastqs_path, el)):
                print(el)
                fastqs = load_txt(os.path.join(fastqs_path, el))
                fast5s_new = os.path.join(fast5s_path, el.rsplit('.', 1)[0])

                for i in tqdm(range(0, len(fastqs), 4)):
                    fastq_to_fast5(i, fastqs, fast5s_new)
        

if __name__ == "__main__":
    main()
