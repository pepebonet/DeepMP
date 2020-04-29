import os
import glob
import h5py
import click
import functools
from multiprocessing import Pool
from tqdm import tqdm

import utils as ut


def rename(filename, pattern='/Analyses/Basecall_1D_001', 
    replacement='/Analyses/Basecall_1D_000'):
    with h5py.File(filename, 'r+') as f:
        try:
            f[replacement] = f[pattern]
            del f[pattern]
            return 1, 0
        except:
            return 0, 1

def rename_multiprocess(reads, pattern, replacement, cpus):
    f = functools.partial(rename, pattern=pattern, replacement=replacement) 

    error = 0; success = 0
    with Pool(cpus) as p: 
        for i, rval in enumerate(p.imap_unordered(f, tqdm(reads))):
            success += rval[0]; error += rval[1]

    print('{} reads failed and {} were succesfully renamed'.format(error, success))


def get_fastq(read, group, subgroup):
    try:
        with h5py.File(read, 'r+') as f:
            return f['/Analyses/{}/{}/Fastq'.format(group, subgroup)]\
                .value.decode()
    except:
        return ''


def get_fastqs_multi(reads, basecall_g, basecall_sg, output, cpus):
    fastqs = ''
    f = functools.partial(get_fastq, group=basecall_g, subgroup=basecall_sg) 
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, tqdm(reads))):
            fastqs += rval
            
    fastqs_sep = fastqs.split('\n')[:-1]
    out_file = os.path.join(output, '{}_{}.fastq'.format(basecall_g, basecall_sg))
    ut._write_list_to_file(out_file, fastqs_sep)


@click.command(short_help='Parser of fast5 files')
@click.argument('input')
@click.option(
    '-rf5', '--rename-fast5', default=False, 
    help='Whther to rename fast5s basecall directory'
)
@click.option(
    '-p', '--pattern', default='/Analyses/Basecall_1D_001', 
    help='Name to replace'
)
@click.option(
    '-r', '--replacement', default='/Analyses/Basecall_1D_000',  
    help='Name to include'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of processes to be used, default 1'
)
@click.option(
    '-ff5', '--fastq-fast5', default=False, 
    help='Whether to obtain fastqs from the fast5s to new file'
)
@click.option(
    '-bs', '--basecall-group', default='Basecall_1D_000',  
    help='Corrected group of fast5 files. default Basecall_1D_000'
)
@click.option(
    '-bs', '--basecall-subgroup', default='BaseCalled_template',  
    help='Corrected subgroup of fast5 files. default BaseCalled_template'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(input, rename_fast5, pattern, replacement, cpus, fastq_fast5, output,
    basecall_group, basecall_subgroup):
    path = os.path.join(input, '*.fast5')
    reads = glob.glob(path)

    if rename_fast5:
        rename_multiprocess(reads, pattern, replacement, cpus)

    if fastq_fast5:
        get_fastqs_multi(reads, basecall_group, basecall_subgroup, output, cpus)


if __name__ == '__main__':
    main()
