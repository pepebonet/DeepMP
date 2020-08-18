#!/usr/bin/env python3

import os
import shutil
import numpy as np
import pandas as pd
import os, sys, gzip, bz2, re
from collections import defaultdict
from collections import OrderedDict



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
            os.mkdir (tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
        return tmp_dir
    else:
        raise FileNotFoundError('File {} not found'.format(features_path))


def get_reads_in_tmp(features_path, reads_per_file, tmp_dir):
    file_idx = 0; reads_cnt = 0; filenames = set()
    reads = set();  last_reads = dict(); zero_counts = dict()
    with openfile(features_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            rd = l.split()[0]; reads.add(rd)
            import pdb;pdb.set_trace()
            if len(reads) % reads_per_file == 1:
                if rd not in last_reads:
                    file_idx += 1
                    last_reads[rd] = True
                filename = os.path.join(tmp_dir, 'tmp_{}.tsv'.format(file_idx))
                smallfile = open(filename,'a'); filenames.append(filename)
            smallfile.write (l)
    smallfile.close()
    return filenames


def tsv_to_feature(file):
    qual = defaultdict(list) # qulity scores
    mis = defaultdict(int) # mismatches
    mat = defaultdict (int) #matches
    ins = defaultdict(int) # insertions
    dele = defaultdict(int) # deletions
    cov = OrderedDict ()  # coverage
    ins_q = defaultdict(list)
    pos = defaultdict(list) # reference positions
    base = {} # ref base

    with openfile(file) as fh:
        for line in fh:
            ary = line.strip().split()
            test = np.zeros(4)
            if ary[-1] == 'M':
                k = (ary[2], int (ary[-3]))
                aln_mem = [(ary[0],ary[2],int(ary[-3]))] #read, ref, refpos; only store last entry not matching insertion
                test[0] = ord(ary[-4])- 33
                base[k] = ary[-2]
                if (ary[-2] != ary[4]):
                    mis[k] += 1
                else:
                    mat[k] += 1
                import pdb;pdb.set_trace()
            if ary[-1] == 'D':
                k = (ary[2], int(ary[-3]))
                aln_mem = (ary[0],ary[2],int(ary[-3]))
                base[k] = ary[-2]
                dele[k] = dele.get(k,0) + 1
            if ary[-1] == 'I':
                last_k = aln_mem[-1][1],aln_mem[-1][2]
                next_k = (ary[2], last_k[1] + 1)
                if last_k[0] != ary[2]:
                    sys.stderr.write(line.strip())
                ins_k_up = (ary[0], ary[2], last_k[1])
                ins_k_down = (ary[0], ary[2], last_k[1] + 1)
                if not (ins_k_down) in ins_q:
                    ins[next_k] = ins.get(next_k,0) + 1
                    ins_q[ins_k_down].append(ord(ary[-4]) - 33)
                if not (ins_k_up) in ins_q:
                    ins[last_k] = ins.get(last_k,0) + 1
                    ins_q[ins_k_up].append(ord(ary[-4]) - 33)
                import pdb;pdb.set_trace()
        import pdb;pdb.set_trace()

def get_feature(ary):
    feature = np.zeros(4); feature[0] = ord(ary[-4])- 33

    if ary[-1] == 'M':
        if (ary[-2] != ary[4]):
            feature[1] = 1
    if ary[-1] == 'D':
        feature[2] = 1
    if ary[-1] == 'I':
        feature[3] = 1
    
    del ary[1]; del ary[2:5]; del ary[-1] 
    ary.extend(list(feature))

    return ary
    


def tsv_to_feature_2(file):
    output = file.rsplit('.', 1)[0] + '_pre_features.tsv'
    with open(output, 'w') as f:
        with openfile(file) as fh:
            for line in fh:
                ary = line.strip().split()
                if ary[-1] in ['M', 'D', 'I']:
                    feature = get_feature(ary)
                    f.write("\t".join(map(str,feature)) + '\n')



def single_read_errors(features_path, label, motif, output, 
    memory_efficient, reads_per_file):
    tmp_dir = get_tmp_dir(features_path)
    tmp_list = get_reads_in_tmp(features_path, reads_per_file, tmp_dir)
    # tmp_list = [os.path.join(os.path.dirname(features_path), 'tmp/tmp_1.tsv')]
    
    #TODO <JB> multiprocess
    for i in tmp_list:
        tsv_to_feature_2(i)
    
