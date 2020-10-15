#!/usr/bin/env python3

import re
import os
import h5py
import pickle
import fnmatch
import numpy as np
import pandas as pd
import tensorflow as tf


basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
                 'Z': 'Z'}

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
code2base_dna = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
code2base_rna = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N'}

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}

iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}

# ------------------------------------------------------------------------------
# DeepMP
# ------------------------------------------------------------------------------

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def recursive_permute(bases_list):
    if len(bases_list) == 1:
        return bases_list[0]
    elif len(bases_list) == 2:
        pseqs = []
        for fbase in bases_list[0]:
            for sbase in bases_list[1]:
                pseqs.append(fbase + sbase)
        return pseqs
    else:
        pseqs = recursive_permute(bases_list[1:])
        pseq_list = [bases_list[0], pseqs]
        return recursive_permute(pseq_list)


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []

    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

    final_outbases = recursive_permute(outbases)

    return final_outbases


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(',')

    motif_seqs = []

    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)

    return motif_seqs


def get_contig2len(ref_path):
    refseq = DNAReference(ref_path)
    chrom2len = {}
    for contigname in refseq.getcontignames():
        chrom2len[contigname] = len(refseq.getcontigs()[contigname])
    del refseq
    return chrom2len


class DNAReference:
    def __init__(self, reffile):
        self._contignames = []
        self._contigs = {}  # contigname 2 contigseq
        with open(reffile, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        self._contigs[contigname] = contigseq
                        self._contignames.append(contigname)
                    contigname = line.strip()[1:].split(' ')[0]
                    contigseq = ''
                else:
                    # turn to upper case
                    contigseq += line.strip().upper()
            self._contigs[contigname] = contigseq
            self._contignames.append(contigname)

    def getcontigs(self):
        return self._contigs

    def getcontignames(self):
        return self._contignames


def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0):
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            sites.append(i+methyloc_in_motif)
    return sites


def kmer2code(kmer_bytes):
    return [base2code_dna[x] for x in kmer_bytes]


def slice_chunks(l, n):
    for i in range(0, len(l) - n):
        yield l[i:i + n]


# ------------------------------------------------------------------------------
# TRAIN AND CALL MODIFICATIONS
# ------------------------------------------------------------------------------

def get_data_sequence(file, kmer, err_features = False):
    ## preprocess data
    if err_features:
        bases, signal_means, signal_stds, signal_medians, signal_range, \
            signal_lens, base_qual, base_mis, base_ins, base_del, label =load_jm_data(file)
    else:
        bases, signal_means, signal_stds, signal_medians, \
            signal_range, signal_lens, label = load_seq_data(file)

    ## embed bases
    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)

    ## prepare inputs for NNs
    if err_features:
        data = concat_tensors_seq_all(embedded_bases, signal_means, signal_stds, signal_medians,
            signal_range, signal_lens, base_qual, base_mis, base_ins, base_del, kmer)
    else:
        data = concat_tensors_seq(embedded_bases, signal_means, signal_stds, signal_medians,
            signal_range, signal_lens, kmer)

    return data, label


def get_data_errors(file, kmer):
    ## preprocess data
    bases, base_qual, base_mis, base_ins, base_del, label = load_err_read(file)

    ## embed bases
    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)

    ## prepare inputs for NNs
    data = concat_tensors_err(embedded_bases, base_qual, base_mis, base_ins, base_del, kmer)

    return data, label


def get_data_jm(file, kmer):
    ## preprocess data
    bases, signal_means, signal_stds, signal_medians, signal_range, \
        signal_lens, base_qual, base_mis, base_ins, base_del, label = load_jm_data(file)

    ## embed bases
    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)

    ## prepare inputs for NNs
    data_sequence = concat_tensors_seq(embedded_bases, signal_means, signal_stds,
                                        signal_medians, signal_range, signal_lens, kmer)
    data_errors = concat_tensors_err(embedded_bases, base_qual, base_mis, base_ins, base_del, kmer)

    return data_sequence, data_errors, label


def concat_tensors_seq(bases, signal_means, signal_stds, signal_medians,
                        signal_range, signal_lens, kmer):
    return tf.concat([bases,
                                tf.reshape(signal_means, [-1, kmer, 1]),
                                tf.reshape(signal_stds, [-1, kmer, 1]),
                                tf.reshape(signal_medians, [-1, kmer, 1]),
                                tf.reshape(signal_range, [-1, kmer, 1]),
                                tf.reshape(signal_lens, [-1, kmer, 1])],
                                axis=2)

def concat_tensors_seq_all(bases, signal_means, signal_stds, signal_medians,
                        signal_range, signal_lens, base_qual, base_mis, base_ins, base_del, kmer):
    return tf.concat([bases,
                                tf.reshape(signal_means, [-1, kmer, 1]),
                                tf.reshape(signal_stds, [-1, kmer, 1]),
                                tf.reshape(signal_medians, [-1, kmer, 1]),
                                tf.reshape(signal_range, [-1, kmer, 1]),
                                tf.reshape(signal_lens, [-1, kmer, 1]),
                                tf.reshape(base_qual, [-1, kmer, 1]),
                                tf.reshape(base_mis, [-1, kmer, 1]),
                                tf.reshape(base_ins, [-1, kmer, 1]),
                                tf.reshape(base_del, [-1, kmer, 1])],
                                axis=2)

def concat_tensors_err(bases, base_qual, base_mis, base_ins, base_del, kmer):
    return tf.concat([bases,
                                tf.reshape(base_qual, [-1, kmer, 1]),
                                tf.reshape(base_mis, [-1, kmer, 1]),
                                tf.reshape(base_ins, [-1, kmer, 1]),
                                tf.reshape(base_del, [-1, kmer, 1])],
                                axis=2)


def load_seq_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_medians = hf['signal_median'][:]
        signal_range = hf['signal_diff'][:]
        signal_lens = hf['signal_lens'][:]
        try:
            label = hf['label'][:]
        except:
            label = hf['methyl_label'][:]

    return bases, signal_means, signal_stds, signal_medians, \
        signal_range, signal_lens, label


def load_jm_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_medians = hf['signal_median'][:]
        signal_range = hf['signal_diff'][:]
        signal_lens = hf['signal_lens'][:]
        base_qual = hf['qual'][:]
        base_mis = hf['mis'][:]
        base_ins = hf['ins'][:]
        base_del = hf['dele'][:]
        label = hf['methyl_label'][:]

    return bases, signal_means, signal_stds, signal_medians, signal_range, \
        signal_lens, base_qual, base_mis, base_ins, base_del, label


def load_error_data(file):

    with h5py.File(file, 'r') as hf:
        X = hf['err_X'][:]
        Y = hf['err_Y'][:]

    return X, Y

def load_err_read(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        base_qual = hf['qual'][:]
        base_mis = hf['mis'][:]
        base_ins = hf['ins'][:]
        base_del = hf['dele'][:]
        label = hf['methyl_label'][:]

    return bases, base_qual, base_mis, base_ins, base_del, label


def load_error_data_kmer(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        X = hf['err_X'][:]
        Y = hf['err_Y'][:]

    return X, Y, bases


def select_columns(df, columns):
    cols = []
    for c in columns.split(','):
        if re.search (r'-',c):
            c1,c2 = c.split('-')
            cols +=  list (range(int(c1), int(c2)+1))
        elif re.search(r':',c):
            c1,c2 = c.split(':')
            cols += list (range(int(c1), int(c2)+1))
        else:
            cols.append(int(c))

    return df[df.columns[cols]]


# ------------------------------------------------------------------------------
# SVM
# ------------------------------------------------------------------------------

def arrange_columns(cols_in):
    cols = []
    for c in cols_in.split(','):
        if re.search (r'-',c):
            c1,c2 = c.split('-')
            cols +=  list (range(int(c1) - 1, int(c2)))
        elif re.search(r':',c):
            c1,c2 = c.split(':')
            cols += list (range(int(c1) - 1, int(c2)))
        else:
            cols.append(int(c) - 1)

    return list(set(cols))


# ------------------------------------------------------------------------------
# OUTPUT FUNCTIONS
# ------------------------------------------------------------------------------

def _write_to_file(file, content, attach=False):
    if attach and os.path.exists(file):
        open_flag = 'a'
    else:
        open_flag = 'w'

    with open(file, open_flag) as f:
        f.write(str(content))


def _write_list_to_file(file, data):
    with open(file, 'w') as f:
        for listitem in data:
            f.write('%s\n' % listitem)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


def save_probs(probs, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save.to_csv(out_probs, sep='\t', index=None)
