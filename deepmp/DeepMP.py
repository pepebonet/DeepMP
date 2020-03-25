import logging
import argparse
from argparse import Namespace

import click

import deepmp.feature_extraction as fe

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# ------------------------------------------------------------------------------
# ARGPARSER
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


@cli.command(short_help='')
def train_NN():
    """Train Neural Network"""
    raise NotImplementedError


@cli.command(short_help='Feature extraction on contexts')
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
    '-nm', '--normalize-method', type=click.Choice(['mad', 'zscore']), default='mad', 
    help='the way for normalizing signals in read level. mad or zscore, default mad'
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
def context_extraction(**kwargs):
    """Perform feature extraction"""

    args = Namespace(**kwargs)
    fe.extract_features(
        args.input, args.reference_path, args.corrected_group, \
        args.basecall_subgroup, args.is_dna, args.motifs, args.cpus, \
        args.positions, args.normalize_method, args.mod_loc, args.kmer_len, \
        args.cent_signals_len, args.methyl_label, args.write_path, args.f5_batch_num
    )
    

if __name__ == '__main__':
    cli()