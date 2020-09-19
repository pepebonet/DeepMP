#!/usr/bin/env python3

import re
import os
import h5py
import pickle
import fnmatch
import numpy as np
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


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []

    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

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

    return recursive_permute(outbases)


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

def get_data_sequence(file, kmer, one_hot=False):
    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_median, signal_lens, label = load_seq_data(file)

    ## embed bases
    if one_hot:
        embedding_size = 5
        embedding_flag += "_one-hot_embedded"
        embedded_bases = tf.one_hot(bases, embedding_size)

    else:
        vocab_size = 1024
        embedding_size = 128
        weight_table = tf.compat.v1.get_variable(
                                "embedding",
                                shape = [vocab_size, embedding_size],
                                dtype=tf.float32,
                                initializer = tf.compat.v1.truncated_normal_initializer(
                                stddev = np.sqrt(2. / vocab_size)
                                ))
        embedded_bases = tf.nn.embedding_lookup(weight_table, bases)

    ## prepare inputs for NNs
    return tf.concat([embedded_bases,
                                    tf.reshape(signal_means, [-1, kmer, 1]),
                                    tf.reshape(signal_stds, [-1, kmer, 1]),
                                    tf.reshape(signal_median, [-1, kmer, 1]),
                                    # tf.reshape(signal_skew, [-1, kmer, 1]),
                                    # tf.reshape(signal_kurt, [-1, kmer, 1]),
                                    # tf.reshape(signal_diff, [-1, kmer, 1]),
                                    tf.reshape(signal_lens, [-1, kmer, 1])],
                                    axis=2), label


def load_seq_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_median = hf['signal_median'][:]
        signal_skew = hf['signal_skew'][:]
        signal_kurt = hf['signal_kurt'][:]
        signal_diff = hf['signal_diff'][:]
        signal_lens = hf['signal_lens'][:]
        label = hf['label'][:]

    return bases, signal_means, signal_stds, signal_median, signal_skew, \
        signal_kurt, signal_diff, signal_lens, label
    # return bases, signal_means, signal_stds, signal_median, signal_lens, label


def load_error_data(file):

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
