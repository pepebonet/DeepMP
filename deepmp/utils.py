#!/usr/bin/env python3
import os
import re
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


def parse_a_line(line):
    def _kmer2code(kmer_bytes):
        return np.array([base2code_dna[x] for x in kmer_bytes.decode("utf-8")], np.int32)

    words = tf.io.decode_csv(line, [[""]] * 12, "\t")
    kmer = tf.py_function(_kmer2code, [words[6]], tf.int32)
    base_mean = tf.strings.to_number(tf.strings.split([words[7]], ",").values, tf.float32)
    base_std = tf.strings.to_number(tf.strings.split([words[8]], ",").values, tf.float32)
    base_signal_len = tf.strings.to_number(tf.strings.split([words[9]], ",").values, tf.int32)
    cent_signals = tf.strings.to_number(tf.strings.split([words[10]], ",").values, tf.float32)
    label = tf.strings.to_number(words[11], tf.int32)

    return kmer, base_mean, base_std, base_signal_len, cent_signals, label


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
