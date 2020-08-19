#!/usr/bin/env python3

import os
import shutil
import numpy as np
import pandas as pd
import os, sys, gzip, bz2, re
from collections import defaultdict
from collections import OrderedDict

from tqdm import tqdm



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
        
        tmp_dir =  os.path.dirname(features_path) + '/tmp1/'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
        return tmp_dir
    else:
        raise FileNotFoundError('File {} not found'.format(features_path))


def get_reads_in_tmp(features_path, reads_per_file, tmp_dir):
    file_idx = 0; reads_cnt = 0; filenames = []
    reads = set();  last_reads = dict(); zero_counts = dict()
    with openfile(features_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            rd = l.split()[0]; reads.add(rd)
            import pdb;pdb.set_trace()
            test = get_feature_2(l)
            # l = all_in_one(l)       
            if len(reads) % reads_per_file == 1:
                if rd not in last_reads:
                    file_idx += 1
                    last_reads[rd] = True
                filename = os.path.join(tmp_dir, 'tmp_{}.tsv'.format(file_idx))
                smallfile = open(filename,'a'); filenames.append(filename)
            if l:
                smallfile.write(l)
    smallfile.close()
    return filenames


def get_reads_in_tmp_2(features_path, reads_per_file, tmp_dir):
    last_read = ''; lines = [];
    tmp_file = os.path.join(tmp_dir, 'tmp_file.tsv')
    with openfile(features_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            rd = l.split()[0]
            if rd != last_read:
                if lines:
                    get_feature_set(lines, last_read, tmp_file)
                last_read = rd; lines = []
            lines.append(l.strip().split())


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
    for k in tqdm(base.keys()):
        Mis = mis[k]; Mat = mat[k]
        Del = dele[k]; Ins = ins[k]; q_lst = qual[k]
        
        inf = map(str, [read, k[0], k[1], base[k], q_lst, Mat, Mis, Ins, Del])
        outh.write(",".join (inf) + '\n')




def all_in_one(line):
    ary = line.strip().split()
    if ary[-1] in ['M', 'D', 'I']:
        return get_feature(ary)
        

def get_feature(ary):
    feature = np.zeros(4); feature[0] = ord(ary[-4])- 33
    #TODO <JB> Fix problem with Insertions!! 
    if ary[-1] == 'M':
        aln_mem = (ary[0],ary[2],int(ary[-3]))
        if (ary[-2] != ary[4]):
            feature[1] = 1
    if ary[-1] == 'D':
        aln_mem = (ary[0],ary[2],int(ary[-3]))
        feature[2] = 1
    if ary[-1] == 'I':
        import pdb;pdb.set_trace()
        feature[3] = 1
    
    del ary[1]; del ary[2:5]; del ary[-1] 
    ary.extend(list(feature))

    return "\t".join(map(str, ary)) + '\n'


def single_read_errors(features_path, label, motif, output, 
    memory_efficient, reads_per_file):
    tmp_dir = get_tmp_dir(features_path)
    tmp_list = get_reads_in_tmp_2(features_path, reads_per_file, tmp_dir)

    # get_feature_2('/workspace/projects/nanopore/stockholm/EpiNano/novoa_features/ecoli/treated/complement/sample.tsv')
    
    #TODO <JB> problem with read name needs to be fixed
    
