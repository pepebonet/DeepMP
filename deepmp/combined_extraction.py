#!/usr/bin/env pyhton3

#!/usr/bin/env python3
import os
import sys
import glob
import time
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from statsmodels import robust
from scipy.stats import kurtosis, skew


import shutil
import functools
import gzip, bz2, re
from itertools import islice
from collections import defaultdict
from collections import OrderedDict

from deepmp import utils as ut
from deepmp.fast5 import Fast5

#TODO <JB, MC> add to parser and delete
reads_group = 'Raw/Reads'
queen_size_border = 2000
time_wait = 5
key_sep = "||"

# ------------------------------------------------------------------------------
# ERROR FUNCTIONS
# ------------------------------------------------------------------------------


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


def get_feature_set(lines):
    qual, mis, mat, ins, dele, ins_q, base = init_params()
    for ary in lines:
        if ary[-1] == 'M':
            mis, mat, qual, base, aln_mem = get_match_mismatch(
                ary, mis, mat, qual, base)
            
        if ary[-1] == 'D':
            dele, base, aln_mem = get_deletions(ary, aln_mem, base, dele) 
            
        if ary[-1] == 'I':
            ins, ins_q = get_insertions(ary, ins, ins_q, aln_mem)
    
    return arrange_features(qual, mis, mat, ins, dele, base)


def arrange_features(qual, mis, mat, ins, dele, base):
    lines = []
    for k in base.keys():
        Mis = mis[k]; Mat = mat[k]
        Del = dele[k]; Ins = ins[k]; q_lst = qual[k]
        lines.append([k[0], k[1], base[k], q_lst, Mat, Mis, Ins, Del])

    return lines


def get_kmer_set(features, kmer_len, motif, mod_loc):
    position = ut.slice_chunks([item[-7] for item in features], kmer_len)
    sequence = ut.slice_chunks([item[-6] for item in features], kmer_len)
    quality = ut.slice_chunks([item[-5] for item in features], kmer_len)
    mismatch = ut.slice_chunks([item[-3] for item in features], kmer_len)
    insertion = ut.slice_chunks([item[-2] for item in features], kmer_len)
    deletion = ut.slice_chunks([item[-1] for item in features], kmer_len)

    loc = int(np.floor(kmer_len / 2))
    motiflen = len(list(motif)[0])

    lines = []
    for pos, seq, qual, mis, ins, dele in zip(
        position, sequence, quality, mismatch, insertion, deletion): 
        if ''.join(seq[loc - mod_loc: loc + motiflen - mod_loc]) in motif:

            pos = pos[loc]; seq = ''.join(seq)
            # qual = ','.join(qual); mis = ','.join(mis)
            # ins = ','.join(ins); dele = ','.join(dele)

            lines.append([pos, features[0][1], seq, qual, mis, ins, dele])
            
    return lines


def openfile(f):
    if f.endswith ('.gz'):
        fh = gzip.open (f,'rt')
    elif f.endswith ('bz') or f.endswith ('bz2'):
        fh = bz2.open(f,'rt')
    else:
        fh = open(f,'rt')
    return fh


def get_error_features(feat_path, kmer_len, motif, mod_loc):
    lines = [];
    with openfile(feat_path) as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            lines.append(l.strip().split())

    features = get_feature_set(lines)
    return get_kmer_set(features, kmer_len, motif, mod_loc)


# ------------------------------------------------------------------------------
# SEQUENCE FUNCTIONS
# ------------------------------------------------------------------------------

def _write_featurestr_to_file(write_fp, featurestr_q):
    with open(write_fp, 'w') as wf:
        while True:
            if featurestr_q.empty():
                time.sleep(time_wait)
                continue
            features_str = featurestr_q.get()
            if features_str == "kill":
                break
            for one_features_str in features_str:
                wf.write(one_features_str + "\n")
            wf.flush()


def _features_to_str(features):
    chrom, pos, alignstrand, loc_in_ref, readname, strand, k_mer, signal_means, \
        signal_stds, signal_median, signal_skew, signal_kurt, signal_diff, \
        signal_lens, cent_signals, qual, mis, ins, dele, methy_label, flag, \
        mid_base, mid_base_min1, mid_base_min2, mid_base_min3, mid_base_min4, \
        mid_base_min5, mid_base_min6, mid_base_min7, mid_base_min8, \
        mid_base_plus1, mid_base_plus2, mid_base_plus3, mid_base_plus4, \
        mid_base_plus5, mid_base_plus6, mid_base_plus7, mid_base_plus8, \
        mid_base_nonorm, mid_base_min1_nonorm, mid_base_min2_nonorm, \
        mid_base_min3_nonorm, mid_base_min4_nonorm, mid_base_min5_nonorm, \
        mid_base_min6_nonorm, mid_base_min7_nonorm, mid_base_min8_nonorm, \
        mid_base_plus1_nonorm, mid_base_plus2_nonorm, mid_base_plus3_nonorm, \
        mid_base_plus4_nonorm, mid_base_plus5_nonorm, mid_base_plus6_nonorm, \
        mid_base_plus7_nonorm, mid_base_plus8_nonorm = features

    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    median_text = ','.join([str(x) for x in np.around(signal_median, decimals=6)])
    skew_text = ','.join([str(x) for x in np.around(signal_skew, decimals=6)])
    kurt_text = ','.join([str(x) for x in np.around(signal_kurt, decimals=6)])
    diff_text = ','.join([str(x) for x in np.around(signal_diff, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    cent_signals_text = ','.join([str(x) for x in cent_signals])
    qual_text = ','.join([str(x) for x in qual])
    mis_text = ','.join([str(x) for x in mis])
    ins_text = ','.join([str(x) for x in ins])
    dele_text = ','.join([str(x) for x in dele])
    mid_base_txt = ','.join([str(x) for x in mid_base])
    mid_base_min1_txt = ','.join([str(x) for x in mid_base_min1])
    mid_base_min2_txt = ','.join([str(x) for x in mid_base_min2])
    mid_base_min3_txt = ','.join([str(x) for x in mid_base_min3])
    mid_base_min4_txt = ','.join([str(x) for x in mid_base_min4])
    mid_base_min5_txt = ','.join([str(x) for x in mid_base_min5])
    mid_base_min6_txt = ','.join([str(x) for x in mid_base_min6])
    mid_base_min7_txt = ','.join([str(x) for x in mid_base_min7])
    mid_base_min8_txt = ','.join([str(x) for x in mid_base_min8])
    mid_base_plus1_txt = ','.join([str(x) for x in mid_base_plus1])
    mid_base_plus2_txt = ','.join([str(x) for x in mid_base_plus2])
    mid_base_plus3_txt = ','.join([str(x) for x in mid_base_plus3])
    mid_base_plus4_txt = ','.join([str(x) for x in mid_base_plus4])
    mid_base_plus5_txt = ','.join([str(x) for x in mid_base_plus5])
    mid_base_plus6_txt = ','.join([str(x) for x in mid_base_plus6])
    mid_base_plus7_txt = ','.join([str(x) for x in mid_base_plus7])
    mid_base_plus8_txt = ','.join([str(x) for x in mid_base_plus8])
    mid_base_nonorm_txt = ','.join([str(x) for x in mid_base_nonorm])
    mid_base_min1_nonorm_txt = ','.join([str(x) for x in mid_base_min1_nonorm])
    mid_base_min2_nonorm_txt = ','.join([str(x) for x in mid_base_min2_nonorm])
    mid_base_min3_nonorm_txt = ','.join([str(x) for x in mid_base_min3_nonorm])
    mid_base_min4_nonorm_txt = ','.join([str(x) for x in mid_base_min4_nonorm])
    mid_base_min5_nonorm_txt = ','.join([str(x) for x in mid_base_min5_nonorm])
    mid_base_min6_nonorm_txt = ','.join([str(x) for x in mid_base_min6_nonorm])
    mid_base_min7_nonorm_txt = ','.join([str(x) for x in mid_base_min7_nonorm])
    mid_base_min8_nonorm_txt = ','.join([str(x) for x in mid_base_min8_nonorm])
    mid_base_plus1_nonorm_txt = ','.join([str(x) for x in mid_base_plus1_nonorm])
    mid_base_plus2_nonorm_txt = ','.join([str(x) for x in mid_base_plus2_nonorm])
    mid_base_plus3_nonorm_txt = ','.join([str(x) for x in mid_base_plus3_nonorm])
    mid_base_plus4_nonorm_txt = ','.join([str(x) for x in mid_base_plus4_nonorm])
    mid_base_plus5_nonorm_txt = ','.join([str(x) for x in mid_base_plus5_nonorm])
    mid_base_plus6_nonorm_txt = ','.join([str(x) for x in mid_base_plus6_nonorm])
    mid_base_plus7_nonorm_txt = ','.join([str(x) for x in mid_base_plus7_nonorm])
    mid_base_plus8_nonorm_txt = ','.join([str(x) for x in mid_base_plus8_nonorm])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, \
        strand, k_mer, means_text, stds_text, median_text, skew_text, \
        kurt_text, diff_text, signal_len_text, cent_signals_text, qual_text, \
        mis_text, ins_text, dele_text, str(methy_label), str(flag)])
        # mid_base_min8_txt, mid_base_min7_txt, mid_base_min6_txt, mid_base_min5_txt, \
        # mid_base_min4_txt, mid_base_min3_txt, mid_base_min2_txt, mid_base_min1_txt, \
        # mid_base_txt, mid_base_plus1_txt, mid_base_plus2_txt, mid_base_plus3_txt, mid_base_plus4_txt, \
        # mid_base_plus5_txt, mid_base_plus6_txt, mid_base_plus7_txt, mid_base_plus8_txt, \
        # mid_base_min8_nonorm_txt, mid_base_min7_nonorm_txt, mid_base_min6_nonorm_txt, mid_base_min5_nonorm_txt, \
        # mid_base_min4_nonorm_txt, mid_base_min3_nonorm_txt, mid_base_min2_nonorm_txt, mid_base_min1_nonorm_txt, \
        # mid_base_nonorm_txt, mid_base_plus1_nonorm_txt, mid_base_plus2_nonorm_txt, mid_base_plus3_nonorm_txt, mid_base_plus4_nonorm_txt, \
        # mid_base_plus5_nonorm_txt, mid_base_plus6_nonorm_txt, mid_base_plus7_nonorm_txt, mid_base_plus8_nonorm_txt])


def _read_position_file(position_file):
    pos_df = pd.read_csv(position_file, sep='\t')
    pos_df['id'] = pos_df['chr'] + '||' + pos_df['start'].astype(str) + '||' + pos_df['strand']
    return pos_df['id'].tolist()


def _fill_files_queue(fast5s_q, fast5_files, batch_size):
    for i in np.arange(0, len(fast5_files), batch_size):
        fast5s_q.put(fast5_files[i:(i+batch_size)])
    return


#Extract signals around methylated base --> Signal Feature Module
def _get_central_signals(signals_list, rawsignal_num=360):
    signal_lens = [len(x) for x in signals_list]

    if sum(signal_lens) < rawsignal_num:
        real_signals = np.concatenate(signals_list)
        cent_signals = np.append(
            real_signals, np.array([0] * (rawsignal_num - len(real_signals)))
        )
    else:
        mid_loc = int((len(signals_list) - 1) / 2)
        mid_base_len = len(signals_list[mid_loc])

        if mid_base_len >= rawsignal_num:
            allcentsignals = signals_list[mid_loc]
            cent_signals = [allcentsignals[x] for x in sorted(
                random.sample(range(len(allcentsignals)), rawsignal_num))]
        else:
            left_len = (rawsignal_num - mid_base_len) // 2
            right_len = rawsignal_num - left_len

            left_signals = np.concatenate(signals_list[:mid_loc])
            right_signals = np.concatenate(signals_list[mid_loc:])

            if left_len > len(left_signals):
                right_len = right_len + left_len - len(left_signals)
                left_len = len(left_signals)
            elif right_len > len(right_signals):
                left_len = left_len + right_len - len(right_signals)
                right_len = len(right_signals)

            assert (right_len + left_len == rawsignal_num)
            if left_len == 0:
                cent_signals = right_signals[:right_len]
            else:
                cent_signals = np.append(
                    left_signals[-left_len:], right_signals[:right_len])

    return cent_signals


def _normalize_signals(signals, normalize_method='mad'):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), np.float(np.std(signals))
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), np.float(robust.mad(signals))
    else:
        raise ValueError('')
    norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


#Raw signal --> Normalization --> alignment --> methylated site --> features
def _extract_features(fast5s, errors, corrected_group, basecall_subgroup, 
    normalize_method, motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
    methy_label, positions, dict_names):
    features_list = []

    error = 0
    for fast5_fp in fast5s:
        try:
            raw_signal = fast5_fp.get_raw_signal()
            norm_signals = _normalize_signals(raw_signal, normalize_method)
            genomeseq, signal_list, signal_nonorm = "", [], []

            events = fast5_fp.get_events(corrected_group, basecall_subgroup)
            for e in events:
                genomeseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])
                signal_nonorm.append(raw_signal[e[0]:(e[0] + e[1])])
            readname = fast5_fp.file.rsplit('/', 1)[1]

            strand, alignstrand, chrom, chrom_start = fast5_fp.get_alignment_info(
                corrected_group, basecall_subgroup
            )

            #TODO CLEAN UP AND PUT IT IN THE PARSER FOR THE DIFFERENT OPTIONS OR ARMONIZE PRIOR TO EXTRACTION
            if dict_names:
                try:
                    error_read = os.path.join(
                        errors, '{}.txt'.format(dict_names[readname.split('.')[0]])
                    )
                except: 
                    try:
                        error_read = os.path.join(
                            errors, '{}.txt'.format(dict_names[readname])
                        )
                    except:
                        error_read = os.path.join(
                            errors, '{}.txt'.format(dict_names[readname.split('.')[0].rsplit('_', 1)[0]])
                        )
                
                error_features = get_error_features(
                    error_read, kmer_len, motif_seqs, methyloc
                )
            else:
                error_read = os.path.join(
                    errors, '{}.txt'.format(readname.rsplit('.', 1)[0])
                )
                
                error_features = get_error_features(
                    error_read, kmer_len, motif_seqs, methyloc
                )
                # raise NotImplementedError('not sure how to deal with it')

            chromlen = chrom2len[chrom]
            if alignstrand == '+':
                chrom_start_in_alignstrand = chrom_start
            else:
                chrom_start_in_alignstrand = \
                    chromlen - (chrom_start + len(genomeseq))

            tsite_locs = ut.get_refloc_of_methysite_in_motif(
                genomeseq, set(motif_seqs), methyloc
            )

            if kmer_len % 2 == 0:
                raise ValueError("kmer_len must be odd")
            num_bases = (kmer_len - 1) // 2

            for loc_in_read in tsite_locs:
                if num_bases <= loc_in_read < len(genomeseq) - num_bases:
                    loc_in_ref = loc_in_read + chrom_start_in_alignstrand

                    if alignstrand == '-':
                        pos = chromlen - 1 - loc_in_ref
                    else:
                        pos = loc_in_ref

                    # if (positions is not None) and (key_sep.join([chrom, \
                    #     str(pos), alignstrand]) not in positions):
                    #     continue
                    if positions is not None: 
                        aa = key_sep.join([chrom, str(pos), alignstrand])
                        if aa in positions: 
                            print('match')
                        else: 
                            continue


                    k_mer = genomeseq[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals = signal_list[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals_nonorm = signal_nonorm[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    
                    signal_lens = [len(x) for x in k_signals]
                    signal_means = [np.mean(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]
                    signal_median = [np.median(x) for x in k_signals]
                    signal_diff = [np.abs(np.max(x) - np.min(x)) for x in k_signals]
                    signal_skew = [skew(x) for x in k_signals]
                    signal_kurtosis = [kurtosis(x) for x in k_signals]

                    cent_signals = _get_central_signals(
                        k_signals, raw_signals_len
                    )

                    mid_loc = int((len(k_signals) - 1) / 2)
                    mid_base = k_signals[mid_loc]
                    mid_base_min1 = k_signals[mid_loc - 1]
                    mid_base_min2 = k_signals[mid_loc - 2]
                    mid_base_min3 = k_signals[mid_loc - 3]
                    mid_base_min4 = k_signals[mid_loc - 4]
                    mid_base_min5 = k_signals[mid_loc - 5]
                    mid_base_min6 = k_signals[mid_loc - 6]
                    mid_base_min7 = k_signals[mid_loc - 7]
                    mid_base_min8 = k_signals[mid_loc - 8]
                    mid_base_plus1 = k_signals[mid_loc + 1]
                    mid_base_plus2 = k_signals[mid_loc + 2]
                    mid_base_plus3 = k_signals[mid_loc + 3]
                    mid_base_plus4 = k_signals[mid_loc + 4]
                    mid_base_plus5 = k_signals[mid_loc + 5]
                    mid_base_plus6 = k_signals[mid_loc + 6]
                    mid_base_plus7 = k_signals[mid_loc + 7]
                    mid_base_plus8 = k_signals[mid_loc + 8]

                    mid_base_nonorm = k_signals_nonorm[mid_loc]
                    mid_base_min1_nonorm = k_signals_nonorm[mid_loc - 1]
                    mid_base_min2_nonorm = k_signals_nonorm[mid_loc - 2]
                    mid_base_min3_nonorm = k_signals_nonorm[mid_loc - 3]
                    mid_base_min4_nonorm = k_signals_nonorm[mid_loc - 4]
                    mid_base_min5_nonorm = k_signals_nonorm[mid_loc - 5]
                    mid_base_min6_nonorm = k_signals_nonorm[mid_loc - 6]
                    mid_base_min7_nonorm = k_signals_nonorm[mid_loc - 7]
                    mid_base_min8_nonorm = k_signals_nonorm[mid_loc - 8]
                    mid_base_plus1_nonorm = k_signals_nonorm[mid_loc + 1]
                    mid_base_plus2_nonorm = k_signals_nonorm[mid_loc + 2]
                    mid_base_plus3_nonorm = k_signals_nonorm[mid_loc + 3]
                    mid_base_plus4_nonorm = k_signals_nonorm[mid_loc + 4]
                    mid_base_plus5_nonorm = k_signals_nonorm[mid_loc + 5]
                    mid_base_plus6_nonorm = k_signals_nonorm[mid_loc + 6]
                    mid_base_plus7_nonorm = k_signals_nonorm[mid_loc + 7]
                    mid_base_plus8_nonorm = k_signals_nonorm[mid_loc + 8]



                    pos_err = [item[0] - 1 for item in error_features]
                    comb_err = error_features[np.argwhere(np.asarray(pos_err) == pos)[0][0]]
                    try: 
                        qual = comb_err[-4]
                        mis = comb_err[-3]
                        ins = comb_err[-2]
                        dele = comb_err[-1]
                    except: 
                        print('Error in the error features occurred...')
                        error += 1
                        continue
                    if np.mean(signal_lens) > 7:
                        flag = 0
                    else:
                        flag = 1
                    features_list.append(
                        (chrom, pos, alignstrand, loc_in_ref, readname, strand,
                        k_mer, signal_means, signal_stds, signal_median,  
                        signal_skew, signal_kurtosis, signal_diff, signal_lens, 
                        cent_signals, qual, mis, ins, dele, methy_label, flag, 
                        mid_base, mid_base_min1, mid_base_min2, mid_base_min3, mid_base_min4,
                        mid_base_min5, mid_base_min6, mid_base_min7, mid_base_min8,
                        mid_base_plus1, mid_base_plus2, mid_base_plus3, mid_base_plus4,
                        mid_base_plus5, mid_base_plus6, mid_base_plus7, mid_base_plus8,
                        mid_base_nonorm, mid_base_min1_nonorm, mid_base_min2_nonorm, 
                        mid_base_min3_nonorm, mid_base_min4_nonorm,
                        mid_base_min5_nonorm, mid_base_min6_nonorm, 
                        mid_base_min7_nonorm, mid_base_min8_nonorm,
                        mid_base_plus1_nonorm, mid_base_plus2_nonorm, 
                        mid_base_plus3_nonorm, mid_base_plus4_nonorm,
                        mid_base_plus5_nonorm, mid_base_plus6_nonorm, 
                        mid_base_plus7_nonorm, mid_base_plus8_nonorm)
                    )
        except Exception:
            error += 1
            continue
        
    return features_list, error


def get_a_batch_features_str(fast5s_q, featurestr_q, errornum_q, err_path,
    corrected_group, basecall_subgroup, normalize_method, motif_seqs, methyloc, 
    chrom2len, kmer_len, raw_signals_len, methy_label, positions, dict_names):
    #Obtain features from every read 
    while not fast5s_q.empty():
        try:
            fast5s = fast5s_q.get()
        except Exception:
            break

        features_list, error_num = _extract_features(
            fast5s, err_path, corrected_group, basecall_subgroup, normalize_method, 
            motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len, 
            methy_label, positions, dict_names
        )
        features_str = []
        for features in features_list:
            features_str.append(_features_to_str(features))

        errornum_q.put(error_num)
        featurestr_q.put(features_str)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)


def find_fast5_files(fast5s_dir, recursive, file_list=None):
    """Find appropriate fast5 files"""
    logging.info("Reading fast5 folder...")
    if file_list:
        fast5s = []
        for el in ut.load_txt(file_list):
            fast5s.append(Fast5(os.path.join(fast5s_dir, el)))
    else:
        if recursive:
            for x in os.walk(fast5s_dir):
                if x[1]:    
                    rec_reads = ([glob.glob(re) for re in \
                        [os.path.join(fast5s_dir, el, '*.fast5') for el in x[1]]])
            fast5s = [Fast5(i) for sub in rec_reads for i in sub]

        else:
            fast5s = [Fast5(os.path.join(fast5s_dir, f)) for f in tqdm(os.listdir(fast5s_dir))
                  if os.path.isfile(os.path.join(fast5s_dir, f))]

    return fast5s


def _extract_preprocess(fast5_dir, motifs, is_dna, reference_path, 
        f5_batch_num, position_file, recursive):
    #Extract list of reads, target motifs and chrom lenghts of the ref genome
    fast5_files = find_fast5_files(fast5_dir, recursive)
    print("{} fast5 files in total".format(len(fast5_files)))

    print("Parsing motifs string...")
    motif_seqs = ut.get_motif_seqs(motifs, is_dna)

    print("Reading genome reference file...")
    chrom2len = ut.get_contig2len(reference_path)

    positions = None
    if position_file is not None:
        print("Reading position file...")
        positions = _read_position_file(position_file)
    
    #Distribute reads into processes
    fast5s_q = mp.Queue()
    _fill_files_queue(fast5s_q, fast5_files, f5_batch_num)

    return motif_seqs, chrom2len, fast5s_q, len(fast5_files), positions


def combine_extraction(fast5_dir, read_errors, ref, cor_g, base_g, dna, motifs,
    nproc, position_file, norm_me, methyloc, kmer_len, raw_sig_len, methy_lab, 
    write_fp, f5_batch_num, recursive, dict_names):

    start = time.time()
    motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = \
        _extract_preprocess(fast5_dir, motifs, dna, ref, f5_batch_num, 
            position_file, recursive
    )

    if dict_names:
        dict_names = ut.load_obj(dict_names)  
        dict_names = {v: k for k, v in dict_names.items()}

    featurestr_q = mp.Queue()
    errornum_q = mp.Queue()

    print('Getting features from nanopore reads...')
    #Start process for feature extraction in every core
    featurestr_procs = []
    if nproc > 1:
        nproc -= 1
    for _ in range(nproc):
        p = mp.Process(
            target=get_a_batch_features_str, args=(fast5s_q, featurestr_q, 
            errornum_q, read_errors, cor_g, base_g, norm_me, motif_seqs, methyloc, chrom2len, 
            kmer_len, raw_sig_len, methy_lab, positions, dict_names)
        )
        p.daemon = True
        p.start()
        featurestr_procs.append(p)

    print("Writing features to file...")
    p_w = mp.Process(
        target=_write_featurestr_to_file, args=(write_fp, featurestr_q)
    )
    p_w.daemon = True 
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in featurestr_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in featurestr_procs:
        p.join() 

    print("finishing the writing process..")
    featurestr_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed..\n"
          "extract_features costs %.1f seconds.." % (errornum_sum, len_fast5s,
                                                     time.time() - start))
