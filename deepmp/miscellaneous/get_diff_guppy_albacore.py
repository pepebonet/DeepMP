#!/usr/bin/envs python3 
import os
import sys
import glob
import click 
import numpy as np
import pandas as pd 
import seaborn as sns
from tqdm import tqdm
from statsmodels import robust
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

sys.path.append('../')
from deepmp import utils as ut
from deepmp.fast5 import Fast5

def find_fast5_files(fast5s_dir):
    """Find appropriate fast5 files"""
    
    return [Fast5(os.path.join(fast5s_dir, f)) \
            for f in tqdm(os.listdir(fast5s_dir)[0:200])
            if os.path.isfile(os.path.join(fast5s_dir, f))]


def _extract_preprocess(fast5_dir, motifs, is_dna, reference_path):
    #Extract list of reads, target motifs and chrom lenghts of the ref genome
    fast5_files = find_fast5_files(fast5_dir)
    print("{} fast5 files in total".format(len(fast5_files)))

    print("Parsing motifs string...")
    motif_seqs = ut.get_motif_seqs(motifs, is_dna)

    print("Reading genome reference file...")
    chrom2len = ut.get_contig2len(reference_path)

    return motif_seqs, chrom2len, fast5_files, len(fast5_files)


def _normalize_signals(signals, normalize_method='mad'):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), np.float(np.std(signals))
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), np.float(robust.mad(signals))
    else:
        raise ValueError('')
    norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def _extract_features(fast5s, corrected_group, basecall_subgroup, 
    normalize_method, motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
    methy_label):
    
    error = 0; feature_list = []; read_features = []

    for fast5 in tqdm(fast5s):
        try:
            raw_signal = fast5.get_raw_signal()
            norm_signals = _normalize_signals(raw_signal, normalize_method)
            genomeseq, signal_list, signal_nonorm = "", [], []

            events = fast5.get_events(corrected_group, basecall_subgroup)
            for e in events:
                genomeseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])
                signal_nonorm.append(raw_signal[e[0]:(e[0] + e[1])])
            readname = fast5.file.rsplit('/', 1)[1]
            
            

            strand, alignstrand, chrom, chrom_start = fast5.get_alignment_info(
                corrected_group, basecall_subgroup
            )
            read_features.append((readname, len(genomeseq), [len(x) for x in signal_list]))
            
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

                    signal_lens = [len(x) for x in k_signals]
                    signal_means = [np.mean(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]
                    signal_median = [np.median(x) for x in k_signals]
                    signal_diff = [np.abs(np.max(x) - np.min(x)) for x in k_signals]
                    signal_skew = [skew(x) for x in k_signals]
                    signal_kurtosis = [kurtosis(x) for x in k_signals]

                    feature_list.append(
                        (chrom, pos, alignstrand, loc_in_ref, readname, strand,
                        k_mer, signal_means, signal_stds, signal_median,  
                        signal_skew, signal_kurtosis, signal_diff, signal_lens, 
                        methy_label)
                    )

        except:
            error += 1 

    return read_features, feature_list, error


def plot_read_features_dist(read_f_000, read_f_001):
    #No real difference in the # of signal distributions
    signals_000 = np.concatenate(read_f_000[2].tolist())
    signals_001 = np.concatenate(read_f_001[2].tolist())

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    sns.displot(x=signals_000, kind="kde", color='blue')
    sns.displot(x=signals_001, kind="kde", color='red')

    plt.tight_layout()
    out_dir = os.path.join('distribution_001.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_gc_features_dist(cgs_f_000, cgs_f_001, output):
    counter = 0
    for i, df in cgs_f_001.groupby(4):
        try:
            sub_000 = cgs_f_000[cgs_f_000[4] == i]
            len_sig_000 = np.concatenate(sub_000[13].tolist())
        except: 
            continue
        
        len_sig_001 = np.concatenate(df[13].tolist())
        guppy_albacore_len = np.concatenate([len_sig_001, len_sig_000])

        guppy = ['Guppy'] * len(len_sig_001)
        albacore = ['Albacore'] * len(len_sig_000)
        guppy_albacore_label = np.concatenate([guppy, albacore])
        guppy_albacore = np.array([guppy_albacore_label, guppy_albacore_len])
        df_len_sig = pd.DataFrame(guppy_albacore.T)
        df_len_sig[1] = df_len_sig[1].astype(int)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

        sns.displot(data=df_len_sig, x=df_len_sig[1], kind="kde", hue=df_len_sig[0])

        plt.tight_layout()
        out_dir = os.path.join(output, 'distribution_cgs_read_{}.png'.format(counter))
        plt.savefig(out_dir)
        plt.close()
        counter += 1


@click.command(short_help='Explore differences guppy and albacore')
@click.option('-r', '--reads', required=True, help='reads to explore')
@click.option('-re', '--reference', required=True, help='reference genome')
@click.option('-o', '--output', required=True, help='Path to save dict')
def main(reads, reference, output):

    motif_seqs, chrom2len, fast5s, len_fast5s = \
        _extract_preprocess(reads, 'GC', True, reference
    )

    corrected_groups = ['RawGenomeCorrected_000', 'RawGenomeCorrected_001']
    for group in corrected_groups:
        read_features, features_list, error_num = _extract_features(
            fast5s, group, 'BaseCalled_template', 'mad', 
            motif_seqs, 0, chrom2len, 17, 360, 1
        )
        if group == 'RawGenomeCorrected_000':
            read_f_000 = pd.DataFrame(read_features)
            cgs_f_000 =  pd.DataFrame(features_list)
        else:
            read_f_001 = pd.DataFrame(read_features)
            cgs_f_001 =  pd.DataFrame(features_list)
        print(error_num)
    n_cgs_000 = len(cgs_f_000) 
    n_cgs_001 = len(cgs_f_001) 

    # plot_read_features_dist(read_f_000, read_f_001)
    plot_gc_features_dist(cgs_f_000, cgs_f_001, output)
    import pdb;pdb.set_trace()




if __name__ == '__main__':
    main()