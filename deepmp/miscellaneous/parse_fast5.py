import os
import glob
import h5py
import click
import functools
from multiprocessing import Pool
from tqdm import tqdm


def rename(filename, pattern='/Analyses/Basecall_1D_001', 
    replacement='/Analyses/Basecall_1D_000'):
    with h5py.File(filename, 'r+') as f:
        try:
            f[replacement] = f[pattern]
            del f[pattern]
            return 1, 0
        except:
            return 0, 1


@click.command(short_help='Parser of fast5 files (Rename hdf5)')
@click.argument('input')
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
def main(input, pattern, replacement, cpus):
    path = os.path.join(input, '*.fast5')
    reads = glob.glob(path)

    f = functools.partial(rename, pattern=pattern, replacement=replacement) 

    error = 0; success = 0
    with Pool(cpus) as p: 
        for i, rval in enumerate(p.imap_unordered(f, tqdm(reads))):
            success += rval[0]; error += rval[1]

    print('{} reads failed and {} were succesfully renamed'.format(error, success))


if __name__ == '__main__':
    main()
