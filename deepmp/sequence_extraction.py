#!/usr/bin/env python3
import os
import sys
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from statsmodels import robust

from deepmp import utils as ut
from deepmp.fast5 import Fast5

#TODO <JB, MC> add to parser and delete
reads_group = 'Raw/Reads'
queen_size_border = 2000
time_wait = 5
key_sep = "||"

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
        signal_stds, signal_lens, cent_signals, methy_label = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    cent_signals_text = ','.join([str(x) for x in cent_signals])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, \
        strand, k_mer, means_text, stds_text, signal_len_text, \
        cent_signals_text, str(methy_label)])


def _read_position_file(position_file):
    postions = set()
    with open(position_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            postions.add(key_sep.join(words[:3]))
    return postions


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
def _extract_features(fast5s, corrected_group, basecall_subgroup, 
    normalize_method, motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
    methy_label, positions):
    features_list = []

    error = 0
    for fast5_fp in fast5s:
        try:
            raw_signal = fast5_fp.get_raw_signal()
            norm_signals = _normalize_signals(raw_signal, normalize_method)
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

                    if (positions is not None) and (key_sep.join([chrom, \
                        str(pos), alignstrand]) not in positions):
                        continue

                    k_mer = genomeseq[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals = signal_list[(
                        loc_in_read - num_bases):(loc_in_read + num_bases + 1)]

                    signal_lens = [len(x) for x in k_signals]

                    signal_means = [np.mean(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]

                    cent_signals = _get_central_signals(
                        k_signals, raw_signals_len
                    )
                    features_list.append(
                        (chrom, pos, alignstrand, loc_in_ref, readname, strand,
                        k_mer, signal_means, signal_stds, signal_lens, 
                        cent_signals, methy_label)
                    )

        except Exception:
            error += 1
            continue

    return features_list, error


def get_a_batch_features_str(fast5s_q, featurestr_q, errornum_q,
    corrected_group, basecall_subgroup, normalize_method, motif_seqs, methyloc, 
    chrom2len, kmer_len, raw_signals_len, methy_label, positions):
    #Obtain features from every read 
    while not fast5s_q.empty():
        try:
            fast5s = fast5s_q.get()
        except Exception:
            break

        features_list, error_num = _extract_features(
            fast5s, corrected_group, basecall_subgroup,normalize_method, 
            motif_seqs, methyloc,chrom2len, kmer_len, raw_signals_len, 
            methy_label, positions
        )
        features_str = []
        for features in features_list:
            features_str.append(_features_to_str(features))

        errornum_q.put(error_num)
        featurestr_q.put(features_str)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)


def find_fast5_files(fast5s_dir, file_list=None):
    """Find appropriate fast5 files"""
    logging.info("Reading fast5 folder...")
    if file_list:
        fast5s = []
        for el in ut.load_txt(file_list):
            fast5s.append(Fast5(os.path.join(fast5s_dir, el)))
    else:
        fast5s = [Fast5(os.path.join(fast5s_dir, f)) for f in tqdm(os.listdir(fast5s_dir))
                  if os.path.isfile(os.path.join(fast5s_dir, f))]

    return fast5s


def _extract_preprocess(fast5_dir, motifs, is_dna, reference_path, 
        f5_batch_num, position_file):
    #Extract list of reads, target motifs and chrom lenghts of the ref genome
    fast5_files = find_fast5_files(fast5_dir)
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


def extract_features(fast5_dir, ref, cor_g, base_g, dna, motifs,
    nproc, position_file, norm_me, methyloc, kmer_len, raw_sig_len, methy_lab, 
    write_fp, f5_batch_num):
    start = time.time()
    motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = \
        _extract_preprocess(fast5_dir, motifs, dna, ref, f5_batch_num, 
            position_file
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
            kmer_len, raw_sig_len, methy_lab, positions)
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
