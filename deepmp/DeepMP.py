#!/usr/bin/env python3

import logging
from argparse import Namespace

import click

from .train import *
from .preprocess import *
from .call_modifications import *
from .pipeline import fast_call
import deepmp.single_read_errors as sre
import deepmp.sequence_extraction as se
import deepmp.combined_extraction as ce



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

@cli.command(short_help='Calling modifications from user end')
@click.option(
    '-m', '--model_type', required=True,
    type=click.Choice(['seq', 'err', 'joint']),
    help='choose model to test'
)
@click.option(
    '-tf', '--test_file', required=True,
    help='path to test set'
)
@click.option(
    '-k', '--kmer', default=17,
    help='kmer length of the sequence'
)
@click.option(
    '-md', '--model_dir', default='',
    help='directory to trained error model'
)
@click.option(
    '-o', '--output', default='',
    help='output path to save files'
)
@click.option(
    '-ef', '--err_features' , is_flag=True,
    help='use error features in sequence model'
)
@click.option(
    '-ut', '--use_threshold' , is_flag=True,
    help='use threshold instead of beta model to call positions'
)
@click.option(
    '-th', '--threshold' , default=0.1,
    help='give a value to the threshold'
)
@click.option(
    '-pos', '--position_test' , is_flag=True,
    help='position analysis'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of processes to be used, default 1'
)
def call_modifications(**kwargs):
    """Call modifications"""
    args = Namespace(**kwargs)

    call_mods_user(
        args.model_type, args.test_file, args.model_dir,
        args.kmer, args.output, args.err_features,
        args.position_test, args.use_threshold, args.threshold, args.cpus
    )

# ------------------------------------------------------------------------------
# FAST CALL FOR JOINT MODEL
# ------------------------------------------------------------------------------
@cli.command(short_help='Fast call modifications pipeline')
@click.option(
    '-f', '--files', required=True,
    help='path to fast5s'
)
@click.option(
    '-ref', '--reference', required=True,
    help='path to test set'
)
@click.option(
    '-md', '--model_dir', required=True,
    help='path to trained model'
)
@click.option(
    '-j', '--jar', required=True,
    help='path to sam2tsv.jar'
)
@click.option(
    '-cpu', '--cpus', default=1, 
    help='number of processes to be used, default 1'
)
@click.option(
    '-fn', '--fix_names', 
    help='path to fix names script to get dict of read names'
)
@click.option(
    '-dt', '--data_type', type=click.Choice(['Ecoli', 'Human']), 
    required=True
    )
def fast_call_joint(**kwargs):
    """fast call modifications"""
    args = Namespace(**kwargs)

    fast_call(
        args.files, args.reference, args.model_dir,
        args.jar, args.cpus, args.fix_names, args.data_type
    )

# ------------------------------------------------------------------------------
# TRAIN NEURAL NETWORKS
# ------------------------------------------------------------------------------

@cli.command(short_help='Trainining neural networks')
@click.option(
    '-m', '--model_type', required=True,
    type=click.Choice(['seq', 'err', 'joint']),
    help='choose model to train'
)
@click.option(
    '-tf', '--train_file', default='',
    help='path to training set'
)
@click.option(
    '-vf', '--val_file', default='',
    help='path to validation set'
)
@click.option(
    '-rnn', '--rnn_type', default='',
    help='rnn type'
)
@click.option(
    '-k', '--kmer', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-ep', '--epochs', default=15,
    help='Number of epochs for training'
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
    '-bs', '--batch_size', default=512,
    help='Batch size for training both models'
)
@click.option(
    '-ef', '--err_features' , is_flag=True,
    help='use error features in sequence model'
)
@click.option(
    '-cp', '--checkpoint', default='',
    help='path to checkpoint file'
)

def train_nns(**kwargs):
    """Train Neural Networks"""
    args = Namespace(**kwargs)

    if args.model_type == 'seq':
        train_sequence(
                args.train_file, args.val_file,
                args.log_dir, args.model_dir, args.batch_size, args.kmer,
                args.epochs, args.err_features, args.rnn_type, args.checkpoint
                )

    elif args.model_type == 'err':
        train_single_error(
                args.train_file, args.val_file,
                args.log_dir, args.model_dir, args.kmer,
                args.epochs, args.batch_size, args.checkpoint
                )

    elif args.model_type == 'joint':
        train_jm(
                args.train_file, args.val_file,
                args.log_dir, args.model_dir, args.batch_size,
                args.kmer, args.epochs, args.checkpoint
                )


# ------------------------------------------------------------------------------
# PREPROCESS DATA
# ------------------------------------------------------------------------------

@cli.command(short_help='Preprocess data for NNs')
@click.option(
    '-ft', '--feature_type', required=True,
    type=click.Choice(['seq', 'err', 'combined']),
    help='which features is the input corresponding to? To the sequence, '
    'to the errors or to both of them (based on the feature extraction). '
    'If choice and files do not correlate errors will rise throughout the script'
)
@click.option(
    '-sf', '--split_features', is_flag=True,
    help='whether to split the features into train test and validation'
)
@click.option(
    '-f', '--features', default='', help='extracted error features'
)
@click.option(
    '-stsv', '--save-tsv', is_flag=True, help='Whether to store tsv. Default = False'
)
@click.option(
    '-pos', '--positions', default='', help='Pass a position list to filter features'
)
@click.option(
    '-st', '--split_type', type=click.Choice(['pos', 'read', 'chr']), default='pos',
    help='Type of train-test-val split to select. Positions or read'
    'pos option creates and independent test set with positions never seen'
    'read option creates and independent test set with reads never seen '
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of processes to be used, default 1'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def preprocess(**kwargs):
    """Preprocess data for training"""

    args = Namespace(**kwargs)

    if args.split_features:
        split_preprocess(
            args.features, args.output, args.save_tsv, args.cpus,
            args.split_type, args.positions, args.feature_type
        )
    else:
        no_split_preprocess(
            args.features, args.output, args.cpus, args.feature_type
        )


# ------------------------------------------------------------------------------
# SINGLE READ ERROR FEATURE EXTRACTION
# ------------------------------------------------------------------------------

@cli.command(short_help='Extract error features per read')
@click.option(
    '-ef', '--error-features', default='',
    help='extracted errors. Folder containing one file for each read'
)
@click.option(
    '-l', '--label', default='1', type=click.Choice(['1', '0']),
)
@click.option(
    '-m', '--motif', default='CG', help='motif of interest'
)
@click.option(
    '-kl', '--kmer_len', default=17, help='len of kmer. default 17'
)
@click.option(
    '--is-dna', '-id', default='yes', help='whether the fast5 files from '
    'DNA sample or RNA. default true, t, yes, 1. set this option to '
    'no/false/0 if the fast5 files are from RNA sample.'
)
@click.option(
    '-mol', '--mod-loc', default=0,
    help='0-based location of the targeted base in the motif, default 0'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of processes to be used, default 1'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def single_read_error_extraction(**kwargs):
    """Perform error feature extraction """

    args = Namespace(**kwargs)

    sre.single_read_errors(
        args.error_features, args.label, args.motif, args.output,
        args.cpus, args.mod_loc,
        args.kmer_len, args.is_dna
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
    'bases, 0 or 1, default 1'
)
@click.option(
    '--is-dna', '-id', default='yes', help='whether the fast5 files from '
    'DNA sample or RNA. default true, t, yes, 1. set this option to '
    'no/false/0 if the fast5 files are from RNA sample.'
)
@click.option(
    '-kl', '--kmer_len', default=17, help='len of kmer. default 17'
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
    '--recursive', '-r', is_flag=True, help='Find reads recursively in subfolders'
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
        args.normalize_method, args.mod_loc, args.kmer_len, \
        args.methyl_label, args.write_path, \
        args.f5_batch_num, args.recursive
    )


# ------------------------------------------------------------------------------
# COMBINED FEATURE EXTRACTION
# ------------------------------------------------------------------------------

@cli.command(short_help='Extraction of sequence features')
@click.option(
    '-fr', '--fast5-reads', required=True,
    help='fast5 reads to extract sequence features'
)
@click.option(
    '-re', '--read-errors', required=True,
    help='path to read errors'
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
    'bases, 0 or 1, default 1'
)
@click.option(
    '--is-dna', '-id', default='yes', help='whether the fast5 files from '
    'DNA sample or RNA. default true, t, yes, 1. set this option to '
    'no/false/0 if the fast5 files are from RNA sample.'
)
@click.option(
    '-kl', '--kmer_len', default=17, help='len of kmer. default 17'
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
    '--recursive', '-r', is_flag=True, help='Find reads recursively in subfolders'
)
@click.option(
    '--dict-names', '-dn', default='', help='Dict to parse read names'
)
@click.option(
    '-o', '--write_path', required=True, help='file path to save the features'
)
def combine_extraction(**kwargs):
    """Perform sequence and error feature extraction"""

    args = Namespace(**kwargs)
    ce.combine_extraction(
        args.fast5_reads, args.read_errors, args.reference_path, args.corrected_group, \
        args.basecall_subgroup, args.is_dna, args.motifs, args.cpus, \
        args.normalize_method, args.mod_loc, args.kmer_len, \
        args.methyl_label, args.write_path, \
        args.f5_batch_num, args.recursive, args.dict_names
    )


if __name__ == '__main__':
    cli()
