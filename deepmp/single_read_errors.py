#!/usr/bin/env python3

import shutil
import functools
import subprocess
import numpy as np
import os, gzip, bz2
from multiprocessing import Pool
from collections import defaultdict

from deepmp import utils as ut

def openfile(f):
    if f.endswith ('.gz'):
        fh = gzip.open (f,'rt')
    elif f.endswith ('bz') or f.endswith ('bz2'):
        fh = bz2.open(f,'rt')
    else:
        fh = open(f,'rt')
    return fh


def get_tmp_dir(features_path):
    if os.path.isfile(features_path):
        
        tmp_dir =  os.path.dirname(features_path) + '/tmp'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
        return tmp_dir
    else:
        raise FileNotFoundError('File {} not found'.format(features_path))


def get_reads_in_tmp(features_path, reads_per_file, tmp_dir):
    file_idx = 0; filenames = set(); reads = set();  last_reads = dict();

    with openfile(features_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue

            rd = l.split()[0]; reads.add(rd)   
            if len(reads) % reads_per_file == 1:
                if rd not in last_reads:
                    file_idx += 1; last_reads[rd] = True

                filename = os.path.join(tmp_dir, 'tmp_{}.tsv'.format(file_idx))
                smallfile = open(filename,'a'); filenames.add(filename)

            smallfile.write(l)
    smallfile.close()

    return filenames


def get_features_from_tmp(feat_path):
    last_read = ''; lines = [];
    out_tmp = feat_path.rsplit('.', 1)[0] + '_freq.tsv'

    with openfile(feat_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            rd = l.split()[0]
            if rd != last_read:
                if lines:
                    get_feature_set(lines, last_read, out_tmp)
                last_read = rd; lines = []
            lines.append(l.strip().split())
    
    if lines:
        get_feature_set(lines, last_read, out_tmp)

    return out_tmp


def get_insertions(ary, ins, ins_q, aln_mem):
    last_k = aln_mem[1],aln_mem[2]
    next_k = (ary[2], last_k[1] + 1)

    ins_k_up = (ary[0], ary[2], last_k[1])
    ins_k_down = (ary[0], ary[2], last_k[1] + 1)
    if not (ins_k_down) in ins_q:
        ins[next_k] = ins.get(next_k,0) + 1
        ins_q[ins_k_down].append(ord(ary[-4]) - 33)
    if not (ins_k_up) in ins_q:
        ins[last_k] = ins.get(last_k,0) + 1
        ins_q[ins_k_up].append(ord(ary[-4]) - 33)
    
    return ins, ins_q


def get_deletions(ary, aln_mem, base, dele):
    k = (ary[2], int(ary[-3]))
    aln_mem = (ary[0],ary[2],int(ary[-3]))
    base[k] = ary[-2]
    dele[k] = dele.get(k,0) + 1
    return dele, base, aln_mem


def get_match_mismatch(ary, mis, mat, qual, base):
    k = (ary[2], int (ary[-3]))
    aln_mem = (ary[0],ary[2],int(ary[-3]))
    qual[k] = ord(ary[-4])- 33
    base[k] = ary[-2]
    if (ary[-2] != ary[4]):
        mis[k] += 1
    else:
        mat[k] += 1
    return mis, mat, qual, base, aln_mem


def init_params():
    return defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), \
        defaultdict(int), defaultdict(list), {}


def get_feature_set(lines, read, out_file):
    qual, mis, mat, ins, dele, ins_q, base = init_params()
    for ary in lines:
        if ary[-1] == 'M':
            mis, mat, qual, base, aln_mem = get_match_mismatch(
                ary, mis, mat, qual, base)
            
        if ary[-1] == 'D':
            dele, base, aln_mem = get_deletions(ary, aln_mem, base, dele) 
            
        if ary[-1] == 'I':
            ins, ins_q = get_insertions(ary, ins, ins_q, aln_mem)

    arrange_output(qual, mis, mat, ins, dele, base, read, out_file)


def arrange_output(qual, mis, mat, ins, dele, base, read, out_file):
    outh = open(out_file, 'a')
    for k in base.keys():
        Mis = mis[k]; Mat = mat[k]
        Del = dele[k]; Ins = ins[k]; q_lst = qual[k]
        
        inf = map(str, [read, k[0], k[1], base[k], q_lst, Mat, Mis, Ins, Del])
        outh.write(",".join (inf) + '\n')


def get_kmer_set(lines, read, out_file, kmer_len, motif, mod_loc, label):
    position = slice_chunks([item[-7] for item in lines], kmer_len)
    sequence = slice_chunks([item[-6] for item in lines], kmer_len)
    quality = slice_chunks([item[-5] for item in lines], kmer_len)
    mismatch = slice_chunks([item[-3] for item in lines], kmer_len)
    insertion = slice_chunks([item[-2] for item in lines], kmer_len)
    deletion = slice_chunks([item[-1] for item in lines], kmer_len)

    loc = int(np.floor(kmer_len / 2))
    motiflen = len(list(motif)[0])

    outh = open(out_file, 'a')

    for pos, seq, qual, mis, ins, dele in zip(
        position, sequence, quality, mismatch, insertion, deletion): 
        if ''.join(seq[loc - mod_loc: loc + motiflen - mod_loc]) in motif:

            pos = pos[loc]; seq = ''.join(seq)
            qual = ','.join(qual); mis = ','.join(mis)
            ins = ','.join(ins); dele = ','.join(dele)

            inf = map(str, [read, pos, lines[0][1], seq, qual, mis, ins, dele, label])
            
            outh.write("\t".join(inf) + '\n')


def get_kmer_features(feat_path, kmer_len, motif, mod_loc, label):
    last_read = ''; lines = [];
    out_tmp = feat_path.rsplit('.', 1)[0] + '_kmer.tsv'
    
    with openfile(feat_path) as fh:
        for l in fh:
            rd = l.split(',')[0]
            if rd != last_read:
                if lines:
                    get_kmer_set(
                        lines, last_read, out_tmp, kmer_len, motif, mod_loc, label
                    )
                last_read = rd; lines = []
            lines.append(l.strip().split(','))

        if lines: 
            get_kmer_set(
                lines, last_read, out_tmp, kmer_len, motif, mod_loc, label
            )


def slice_chunks(l, n):
    for i in range(0, len(l) - n):
        yield l[i:i + n]


def concat_features(tmp_dir, output):
    all_files = os.path.join(tmp_dir, '*_kmer.tsv')
    if output:
        out_file = output
    else:
        out_file = os.path.join(tmp_dir.rsplit('/', 1)[0], 'single_read_errors.tsv')
    cmd = 'cat {} > {}'. format(all_files, out_file)
    subprocess.call(cmd, shell=True)
    subprocess.call('rm -r {}'.format(tmp_dir), shell=True)


def single_read_errors(features_path, label, motifs, output, reads_per_file, 
    cpus, mod_loc, kmer_len, is_dna):

    tmp_dir = get_tmp_dir(features_path)
    
    print("Splitting into several files...")
    tmp_list = get_reads_in_tmp(features_path, reads_per_file, tmp_dir)

    print("Getting error features...")
    out_features = []
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(get_features_from_tmp, tmp_list)):
            out_features.append(rval)

    print("Parsing motifs string...")
    motif_seqs = ut.get_motif_seqs(motifs, is_dna)

    print("Getting kmer error features...")
    f = functools.partial(get_kmer_features, kmer_len=kmer_len, \
            motif=motif_seqs, mod_loc=mod_loc, label=label)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, out_features)):
            pass
    
    print("Concating output...")
    concat_features(tmp_dir, output)