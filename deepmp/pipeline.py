#!/usr/bin/env python3

import os
import glob
import h5py
import functools
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import deepmp.combined_extraction as ce
import deepmp.preprocess as pre
import deepmp.call_modifications as cl

def _write_list_to_file(file, data):
    with open(file, 'w') as f:
        for listitem in data:
            f.write('%s\n' % listitem)

def get_fastq(read, group, subgroup):
    try:
        with h5py.File(read, 'r+') as f:
            return f['/Analyses/{}/{}/Fastq'.format(group, subgroup)][()].decode()
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
    _write_list_to_file(out_file, fastqs_sep)


def fast_call(input, ref, model_path, javafile, cpus, fix_names, data_type):

    reads = glob.glob(os.path.join(input, '*.fast5'))
    get_fastqs_multi(reads, 'Basecall_1D_000', 'BaseCalled_template', './', 24)

    print("mapping to the reference genome...")
    cmd1 = ['minimap2', '-ax', 'map-ont', ref, 'Basecall_1D_000_BaseCalled_template.fastq' ]
    map = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
    cmd2 = ['samtools', 'view','-hSb']
    sam1 = subprocess.Popen(cmd2, stdin=map.stdout, stdout=subprocess.PIPE)
    cmd3 = ['samtools', 'sort', '-@', cpus, '-o', 'sample.bam']
    sam2 = subprocess.Popen(cmd3, stdin=sam1.stdout)
    ## wait for the command to complete
    sam2.communicate()
    map.stdout.close()
    sam1.stdout.close()
    
    print("indexing...")
    subprocess.run(['samtools', 'index', 'sample.bam'])
    print('calling variants...')
    cmd4 = ['samtools', 'view', '-h','-F', '3844', 'sample.bam']
    sam3 = subprocess.Popen(cmd4, stdout=subprocess.PIPE)
    cmd5 = ['java', '-jar', javafile, '-r', ref]
    with open('sample.tsv','wb') as output:
        ja =subprocess.Popen(cmd5, stdin=sam3.stdout, stdout=output)
    ja.communicate()
    sam3.stdout.close()

    
    cmd6 = ['python', fix_names, '-dt', data_type, '-f',  \
        'Basecall_1D_000_BaseCalled_template.fastq', '-o', 'dict_reads.pkl']
    dic = subprocess.Popen(cmd6, stdout=subprocess.PIPE)
    dic.communicate()

    print('preparing for feature extraction...')
    subprocess.run(['mkdir', 'tmp'])
    sep = subprocess.Popen(["""awk 'NR==1{ h=$0 }NR>1{ print (!a[$2]++? h ORS $0 : $0) > "tmp/"$1".txt" }' sample.tsv"""],shell=True)
    sep.communicate()

    print('running feature extraction...')
    ce.combine_extraction(input, 'tmp/', ref, 'RawGenomeCorrected_000',
                            'BaseCalled_template', 'yes', 'CG', cpus, 'mad',
                            0, 17, '1', './features.tsv', 100, False, 'dict_reads.pkl')

    print('writing h5...')
    pre.no_split_preprocess('features.tsv', '.', cpus,'combined')

    cl.call_mods_user('joint', 'test/', model_path,
            17, './', False, True, False, 0.1, cpus)

    return None
