#!/usr/bin/env python3
import os
import glob
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from statsmodels import robust
from scipy.stats import kurtosis, skew

from deepmp import utils as ut
from deepmp.fast5 import Fast5
from deepmp import combined_extraction as ce

queen_size_border = 2000
time_wait = 5

# ------------------------------------------------------------------------------
# FUNCTIONS
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
        signal_stds, signal_median, signal_diff, methy_label = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    median_text = ','.join([str(x) for x in np.around(signal_median, decimals=6)])
    diff_text = ','.join([str(x) for x in np.around(signal_diff, decimals=6)])
    
    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, \
        strand, k_mer, means_text, stds_text, median_text, \
        diff_text, str(methy_label)])


#Raw signal --> Normalization --> alignment --> methylated site --> features
def _extract_features(fast5s, corrected_group, basecall_subgroup, 
    normalize_method, motif_seqs, methyloc, chrom2len, kmer_len, methy_label):
    features_list = []

    error = 0
    for fast5_fp in fast5s:
        try:
            raw_signal = fast5_fp.get_raw_signal()
            norm_signals = ce._normalize_signals(raw_signal, normalize_method)
            genomeseq, signal_list = "", []

            events = fast5_fp.get_events(corrected_group, basecall_subgroup)
            for e in events:
                genomeseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])
            readname = fast5_fp.file.rsplit('/', 1)[1]

            strand, alignstrand, chrom, chrom_start = fast5_fp.get_alignment_info(
                corrected_group, basecall_subgroup
            )

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

                    k_mer = genomeseq[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals = signal_list[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]

                    signal_means = [np.mean(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]
                    signal_median = [np.median(x) for x in k_signals]
                    signal_diff = [np.abs(np.max(x) - np.min(x)) for x in k_signals]
                    
                    features_list.append(
                        (chrom, pos, alignstrand, loc_in_ref, readname, strand,
                        k_mer, signal_means, signal_stds, signal_median,  
                        signal_diff, methy_label)
                    )
                    
        except Exception:
            error += 1
            continue

    return features_list, error


def get_a_batch_features_str(fast5s_q, featurestr_q, errornum_q,
    corrected_group, basecall_subgroup, normalize_method, motif_seqs, methyloc, 
    chrom2len, kmer_len, methy_label):
    #Obtain features from every read 
    while not fast5s_q.empty():
        try:
            fast5s = fast5s_q.get()
        except Exception:
            break

        features_list, error_num = _extract_features(
            fast5s, corrected_group, basecall_subgroup,normalize_method, 
            motif_seqs, methyloc,chrom2len, kmer_len, 
            methy_label
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
        f5_batch_num, recursive):
    #Extract list of reads, target motifs and chrom lenghts of the ref genome
    fast5_files = find_fast5_files(fast5_dir, recursive)
    print("{} fast5 files in total".format(len(fast5_files)))

    print("Parsing motifs string...")
    motif_seqs = ut.get_motif_seqs(motifs, is_dna)

    print("Reading genome reference file...")
    chrom2len = ut.get_contig2len(reference_path)

    #Distribute reads into processes
    fast5s_q = mp.Queue()
    ce._fill_files_queue(fast5s_q, fast5_files, f5_batch_num)

    return motif_seqs, chrom2len, fast5s_q, len(fast5_files)


def extract_features(fast5_dir, ref, cor_g, base_g, dna, motifs,
    nproc, norm_me, methyloc, kmer_len, methy_lab, 
    write_fp, f5_batch_num, recursive):
    start = time.time()
    motif_seqs, chrom2len, fast5s_q, len_fast5s = \
        _extract_preprocess(fast5_dir, motifs, dna, ref, f5_batch_num, 
            recursive
    )
    
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
            errornum_q, cor_g, base_g, norm_me, motif_seqs, methyloc, chrom2len, 
            kmer_len, methy_lab)
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
