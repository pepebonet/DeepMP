import argparse
from argparse import Namespace

import deepmp.feature_extraction as fe


# ------------------------------------------------------------------------------
# ARGPARSER
# ------------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
          description='*** Feature extraction of fast5 nanopore data ***'
    )
    parser.add_argument(
        'input', help='Absolute or relative path to fast5 nanopore data. '
    )
    parser.add_argument(
        '--recursively', '-r', action='store', type=str, required=False,
        default='yes', help='is to find fast5 files from fast5_dir recursively. '
        'default true, t, yes, 1'
    )
    parser.add_argument(
        '--corrected_group', '-cg', action='store', type=str, required=False,
        default='RawGenomeCorrected_000', help='the corrected_group of '
        'fast5 files after tombo re-squiggle. default RawGenomeCorrected_000'
    )
    parser.add_argument(
        '--basecall_subgroup', '-bs', action='store', type=str, required=False,
        default='BaseCalled_template', help='Corrected subgroup of '
        'fast5 files. default BaseCalled_template'
    )
    parser.add_argument(
        '--reference_path', '-rp', action='store', type=str, required=True,
        help='Reference genome to be used. .fa file'
    )
    parser.add_argument(
        '--is_dna', '-id', action='store', type=str, 
        required=False, default='yes', help='whether the fast5 files from '
        'DNA sample or RNA. default true, t, yes, 1. set this option to '
        'no/false/0 if the fast5 files are from RNA sample.'
    )
    parser.add_argument(
        '--normalize_method', '-nm', action='store', type=str, 
        choices=['mad', 'zscore'], default='mad', required=False, 
        help='the way for normalizing signals in read level. mad or zscore, '
        'default mad'
    )
    parser.add_argument(
        '--methy_label', '-ml', action='store', type=int, choices=[1, 0], 
        required=False, default=1, help='the label of the interested modified '
        'bases, this is for training. 0 or 1, default 1'
    )
    parser.add_argument(
        '--kmer_len', '-kl', action='store', type=int, required=False, default=17,
       help='len of kmer. default 17'
    )
    parser.add_argument(
        '--cent_signals_len', '-csl', action='store', type=int, required=False, 
        default=360, help='the number of signals to be used in deepsignal, '
        'default 360'
    )
    parser.add_argument(
        '--motifs', '-m', action='store', type=str,
        required=False, default='G', help='motif seq to be extracted, default:G.'
    )
    parser.add_argument(
        '--mod_loc', '-mol', action='store', type=int, required=False, default=0,
        help='0-based location of the targeted base in the motif, default 0'
    )
    parser.add_argument(
        '--positions', '-p', action='store', type=str,
        required=False, default=None, help='Tap delimited file with a list of '
        'positions. default None'
    )
    parser.add_argument(
        '--write_path', '-o', action='store', type=str, required=True,
        help='file path to save the features'
    )
    parser.add_argument(
        '--nproc', '-np', action='store', type=int, default=1, required=False,
        help='number of processes to be used, default 1'
    )
    parser.add_argument(
        '--f5_batch_num', '-bn', action='store', type=int, 
        default=100, required=False, help='number of files to be processed by '
        'each process one time, default 100'
    )

    args = parser.parse_args()

    context_extraction(args)

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------

def context_extraction(args):
    fe.extract_features(
        args.input, args.reference_path, args.recursively, args.corrected_group, \
        args.basecall_subgroup, args.is_dna, args.motifs, args.f5_batch_num, \
        args.positions, args.normalize_method, args.mod_loc, args.kmer_len, \
        args.cent_signals_len, args.methy_label, args.write_path
    )


if __name__ == '__main__':
    parse_args()