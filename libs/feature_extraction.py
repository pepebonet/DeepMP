#!/usr/bin/env python3
import sys
import argparse
import time
import h5py
import random
import numpy as np

import utils as ut
import multiprocessing as mp
from statsmodels import robust

reads_group = 'Raw/Reads'
queen_size_border = 2000
time_wait = 5
key_sep = "||"

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

    return args


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


def _get_alignment_attrs_of_each_strand(strand_path, h5obj):
    strand_basecall_group_alignment = h5obj['/'.join([strand_path, 'Alignment'])]
    alignment_attrs = strand_basecall_group_alignment.attrs

    if strand_path.endswith('template'):
        strand = 't'
    else:
        strand = 'c'
    if sys.version_info[0] >= 3:
        try:
            alignstrand = str(alignment_attrs['mapped_strand'], 'utf-8')
            chrom = str(alignment_attrs['mapped_chrom'], 'utf-8')
        except TypeError:
            alignstrand = str(alignment_attrs['mapped_strand'])
            chrom = str(alignment_attrs['mapped_chrom'])
    else:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = str(alignment_attrs['mapped_chrom'])
    chrom_start = alignment_attrs['mapped_start']

    return strand, alignstrand, chrom, chrom_start


def _get_readid_from_fast5(h5file):
    first_read = list(h5file[reads_group].keys())[0]
    if sys.version_info[0] >= 3:
        try:
            read_id = str(
                h5file['/'.join(
                    [reads_group, first_read])].attrs['read_id'], 'utf-8')
        except TypeError:
            read_id = str(
                h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    else:
        read_id = str(
            h5file['/'.join([reads_group, first_read])].attrs['read_id'])

    return read_id


def _get_alignment_info_fast5(fast5_path, corrected_group='RawGenomeCorrected_000', 
    basecall_subgroup='BaseCalled_template'):
    try:
        h5file = h5py.File(fast5_path, mode='r')
        corrgroup_path = '/'.join(['Analyses', corrected_group])

        if '/'.join([corrgroup_path, basecall_subgroup, 'Alignment']) in h5file:
            # fileprefix = os.path.basename(fast5_path).split('.fast5')[0]
            readname = _get_readid_from_fast5(h5file)
            strand, alignstrand, chrom, chrom_start = \
                _get_alignment_attrs_of_each_strand(
                    '/'.join([corrgroup_path, basecall_subgroup]), h5file
            )
            h5file.close()
            return readname, strand, alignstrand, chrom, chrom_start
        else:
            return '', '', '', '', ''
    except IOError:
        print("the {} can't be opened".format(fast5_path))
        return '', '', '', '', ''


def _get_label_raw(fast5_fn, correct_group, correct_subgroup):
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_dat = list(fast5_data[reads_group].values())[0]
        raw_dat = raw_dat['Signal'][()]
    except Exception:
        raise RuntimeError('Raw data is not stored in Raw/Reads/Read_[read#] so '
                           'new segments cannot be identified.')

    # Get Events
    try:
        event = fast5_data['/Analyses/'+correct_group + '/' + correct_subgroup + '/Events']
        corr_attrs = dict(list(event.attrs.items()))
    except Exception:
        raise RuntimeError('events not found.')

    read_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']

    starts = list(map(lambda x: x+read_start_rel_to_raw, event['start']))
    lengths = event['length'].astype(np.int)
    base = [x.decode("UTF-8") for x in event['base']]
    assert len(starts) == len(lengths)
    assert len(lengths) == len(base)
    events = list(zip(starts, lengths, base))
    return raw_dat, events


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
            raw_signal, events = _get_label_raw(
                fast5_fp, corrected_group, basecall_subgroup
            )
            norm_signals = _normalize_signals(raw_signal, normalize_method)
            genomeseq, signal_list = "", []
            for e in events:
                genomeseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])

            readname, strand, alignstrand, chrom, chrom_start = \
                _get_alignment_info_fast5(
                    fast5_fp, corrected_group, basecall_subgroup
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
            # print(tsite_locs)
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
                    # print(cent_signals, k_signals)
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


def _extract_preprocess(fast5_dir, is_recursive, motifs, is_dna, reference_path, 
        f5_batch_num, position_file):
    #Extract list of reads, target motifs and chrom lenghts of the ref genome
    fast5_files = ut.get_fast5s(fast5_dir, is_recursive)
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


def extract_features(fast5_dir, ref, recursive, cor_g, base_g, dna, motifs,
    nproc, position_file, norm_me, methyloc, kmer_len, raw_sig_len, methy_lab, 
    write_fp):
    start = time.time()
    motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = \
        _extract_preprocess(fast5_dir, recursive, motifs, dna, ref, nproc, 
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


def main(args):
    extract_features(
        args.input, args.reference_path, args.recursively, args.corrected_group, \
        args.basecall_subgroup, args.is_dna, args.motifs, args.f5_batch_num, \
        args.positions, args.normalize_method, args.mod_loc, args.kmer_len, \
        args.cent_signals_len, args.methy_label, args.write_path
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)