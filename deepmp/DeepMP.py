import logging
import argparse
from argparse import Namespace

import click

from .train import *
from .preprocess import *
import deepmp.sequence_extraction as se
import deepmp.error_extraction as ee

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# ------------------------------------------------------------------------------
# CLICK
# ------------------------------------------------------------------------------

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--debug', help="Show more progress details", is_flag=True)
def cli(debug):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level)

    if not debug:
        # Hide bgdata messages
        logging.getLogger('bgdata').setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# CALL MODIFICATIONS
# ------------------------------------------------------------------------------

@cli.command(short_help='Calling modifications')
def call_modifications():
    """Call modifications"""

    raise NotImplementedError


# ------------------------------------------------------------------------------
# TRAIN NEURAL NETWORKS
# ------------------------------------------------------------------------------

#TODO <MC,PB> An additional parser might be needed in train.py
#TODO <MC,PB> Separate each NN module?
#TODO <MC, PB> How to combine outputs properly --> Joint model
@cli.command(short_help='Trainining neural networks')
@click.option(
    '-ts', '--train_sequence', default='',
    help='path to sequence features for training'
)
@click.option(
    '-vs', '--val_sequence', default='',
    help='path to sequence features for validation'
)
@click.option(
    '-one_hot', '--one_hot_embedding', is_flag=True,
    help='use one hot embedding'
)
@click.option(
    '-rnn', '--rnn_type', default='',
    help='rnn type'
)
@click.option(
    '-ks', '--kmer_sequence', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-es', '--epochs_sequence', default=10,
    help='Number of epochs for sequence training'
)
@click.option(
    '-md', '--model_dir', default='models/',
    help='directory to trained model'
)
@click.option(
    '-ld', '--log_dir', default='logs/',
    help='training log directory'
)
@click.option(
    '-te', '--train_errors', default='',
    help='path to error features for training'
)
@click.option(
    '-ve', '--val_errors', default='',
    help='path to error features for validation'
)
@click.option(
    '-fe', '--features_errors', default=20,
    help='Number of error features to select'
)
@click.option(
    '-ee', '--epochs_errors', default=20,
    help='Number of epochs for error training'
)
@click.option(
    '-bs', '--batch_size', default=512,
    help='Batch size for training both models'
)
def train_nns(**kwargs):
    """Train Neural Networks"""
    args = Namespace(**kwargs)

    if args.train_sequence:
        train_sequence(
                args.train_sequence, args.val_sequence,
                args.log_dir, args.model_dir, args.batch_size, args.kmer_sequence,
                args.epochs_sequence, args.one_hot_embedding, args.rnn_type,
                )

    if args.train_errors:
        train_errors(
                args.train_errors, args.val_errors,
                args.log_dir, args.model_dir, args.features_errors,
                args.epochs_errors, args.batch_size
                )


# ------------------------------------------------------------------------------
# MERGE & PREPROCESS DATA
# ------------------------------------------------------------------------------

@cli.command(short_help='Merge features and preprocess data for NNs')
@click.option(
    '-ft', '--feature_type', required=True,
    type=click.Choice(['seq', 'err', 'both']),
    help='which features is the input corresponding to? To the sequence, '
    'to the errors or to both of them. If choice and files do not correlate '
    'errors will rise throughout the script'
)
@click.option(
    '-et', '--error-treated', default='', help='extracted error features'
)
@click.option(
    '-eu', '--error-untreated', default='', help='extracted error features'
)
@click.option(
    '-st', '--sequence-treated', default='', help='extracted sequence features'
)
@click.option(
    '-su', '--sequence-untreated', default='', help='extracted sequence features'
)
@click.option(
    '-nef', '--num-err-feat', default=20, help='# Error features to select'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def merge_and_preprocess(feature_type, error_treated, error_untreated, 
    sequence_treated, sequence_untreated, num_err_feat, output):
    if feature_type == 'both':
        do_seq_err_preprocess(
            sequence_treated, sequence_untreated, error_treated, 
            error_untreated, output, num_err_feat
        )
    else:
        do_single_preprocess(
            feature_type, sequence_treated, sequence_untreated, 
            error_treated, error_untreated, output, num_err_feat
        )


# ------------------------------------------------------------------------------
# ERROR FEATURE EXTRACTION
# ------------------------------------------------------------------------------

@cli.command(short_help='Extract error features after Epinano pipeline')
@click.option(
    '-ef', '--error-features', default='', 
    help='extracted error through epinano pipeline'
)
@click.option(
    '-l', '--label', default='1', type=click.Choice(['1', '0']),
)
@click.option(
    '-m', '--motif', default='CG', help='motif of interest'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def error_sequence_extraction(**kwargs):
    """Perform error feature extraction """

    args = Namespace(**kwargs)
    ee.process_error_features(
        args.error_features, args.label, args.motif, args.output
    )


# ------------------------------------------------------------------------------
# SEQUENCE FEATURE EXTRACTION
# ------------------------------------------------------------------------------

@cli.command(short_help='Extraction of sequence features')
@click.argument(
    'input'
)
@click.option(
    '-rp', '--reference-path', required=True,
    help='Reference genome to be used. .fa file'
)
@click.option(
    '-cg', '--corrected-group', default='RawGenomeCorrected_000',
    help='the corrected_group of fast5 files after tombo re-squiggle. '
    'default RawGenomeCorrected_000'
)
@click.option(
    '-bs', '--basecall-subgroup', default='BaseCalled_template',
    help='Corrected subgroup of fast5 files. default BaseCalled_template'
)
@click.option(
    '-nm', '--normalize-method', type=click.Choice(['mad', 'zscore']),
    default='mad', help='the way for normalizing signals in read level. '
    'mad or zscore, default mad'
)
@click.option(
    '--methyl-label', '-ml', type=click.Choice(['1', '0']),
    default='1', help='the label of the interested modified '
    'bases, this is for training. 0 or 1, default 1'
)
@click.option(
    '--is-dna', '-id', default='yes', help='whether the fast5 files from '
    'DNA sample or RNA. default true, t, yes, 1. set this option to '
    'no/false/0 if the fast5 files are from RNA sample.'
)
@click.option(
    '-kl', '--kmer_len', default=17, help='len of kmer. default 1'
)
@click.option(
    '-m', '--motifs', default='G', help='motif seq to be extracted, default:G.'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of processes to be used, default 1'
)
@click.option(
    '--f5-batch-num', '-bn', default=100,
    help='number of files to be processed by each process one time, default 100'
)
@click.option(
    '--mod-loc', '-mol', default=0,
    help='0-based location of the targeted base in the motif, default 0'
)
@click.option(
    '--positions', '-p', default=None,
    help='Tap delimited file with a list of positions. default None'
)
@click.option(
    '--cent-signals-len', '-csl', default=360,
    help='the number of signals to be used in deepsignal, '
    'default 360'
)
@click.option(
    '-o', '--write_path', required=True, help='file path to save the features'
)
def sequence_feature_extraction(**kwargs):
    """Perform sequence feature extraction"""

    args = Namespace(**kwargs)
    se.extract_features(
        args.input, args.reference_path, args.corrected_group, \
        args.basecall_subgroup, args.is_dna, args.motifs, args.cpus, \
        args.positions, args.normalize_method, args.mod_loc, args.kmer_len, \
        args.cent_signals_len, args.methyl_label, args.write_path, \
        args.f5_batch_num
    )


if __name__ == '__main__':
    cli()
