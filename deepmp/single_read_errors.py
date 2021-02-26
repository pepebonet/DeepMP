#!/usr/bin/env python3

import functools
import subprocess
import numpy as np
import os, gzip, bz2
from multiprocessing import Pool

from deepmp import utils as ut
from deepmp import combined_extraction as ce


def get_tmp_dir(output):
    if os.path.isdir(output):
        tmp_dir =  os.path.join(output, 'tmp_errors')
        
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
            return tmp_dir
        else:
            raise ValueError('Folder {} already exists'.format(tmp_dir))
        
    else:
        raise FileNotFoundError('File {} not found'.format(output))


def get_error_features(feat_path, kmer_len, motif, mod_loc, label, tmp_dir):
    lines = [];
    with ut.openfile(feat_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            lines.append(l.strip().split())

    features = ce.get_feature_set(lines)
    read = feat_path.rsplit('/', 1)[1]
    out_file = os.path.join(tmp_dir, read)

    return get_kmer_set(features, read, out_file, kmer_len, motif, mod_loc, label)


def get_kmer_set(lines, read, out_file, kmer_len, motif, mod_loc, label):
    position = ut.slice_chunks([item[-7] for item in lines], kmer_len)
    sequence = ut.slice_chunks([item[-6] for item in lines], kmer_len)
    quality = ut.slice_chunks([item[-5] for item in lines], kmer_len)
    mismatch = ut.slice_chunks([item[-3] for item in lines], kmer_len)
    insertion = ut.slice_chunks([item[-2] for item in lines], kmer_len)
    deletion = ut.slice_chunks([item[-1] for item in lines], kmer_len)

    loc = int(np.floor(kmer_len / 2))
    motiflen = len(list(motif)[0])

    outh = open(out_file, 'a')

    features = []
    for pos, seq, qual, mis, ins, dele in zip(
        position, sequence, quality, mismatch, insertion, deletion): 
        if ''.join(seq[loc - mod_loc: loc + motiflen - mod_loc]) in motif:

            pos = pos[loc]; seq = ''.join(seq)
            qual = ','.join([str(x) for x in qual])
            mis = ','.join([str(x) for x in mis])
            ins = ','.join([str(x) for x in ins])
            dele = ','.join([str(x) for x in dele])
            
            feature = "\t".join([read, str(pos), lines[0][0], seq, qual, \
                mis, ins, dele, label])
            outh.write(feature + '\n')


def concat_features(tmp_dir, output):
    all_files = os.path.join(tmp_dir, '*.txt')
    if output:
        out_file = os.path.join(output, 'single_read_errors.tsv')
    else:
        out_file = os.path.join(tmp_dir.rsplit('/', 1)[0], 'single_read_errors.tsv')

    cmd = 'cat {} > {}'. format(all_files, out_file)
    subprocess.call(cmd, shell=True)
    subprocess.call('rm -r {}'.format(tmp_dir), shell=True)


def single_read_errors(features_path, label, motifs, output, 
    cpus, mod_loc, kmer_len, is_dna):

    tmp_dir = get_tmp_dir(output)
    tmp_list = [os.path.join(features_path, i) for i in os.listdir(features_path)]

    print("Parsing motifs string...")
    motif_seqs = ut.get_motif_seqs(motifs, is_dna)

    print("Getting error features...")
    f = functools.partial(get_error_features, kmer_len=kmer_len, \
            motif=motif_seqs, mod_loc=mod_loc, label=label, tmp_dir=tmp_dir)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, tmp_list)):
            pass

    print("Concating output...")
    concat_features(tmp_dir, output)