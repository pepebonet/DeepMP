import logging
import sys

import h5py
import numpy as np


class Fast5:

    READS_GROUP = 'Raw/Reads'

    def __init__(self, path):
        self.file = path
        self._read = path.rsplit('/', 1)[1]
        self._read_id = path.rsplit('.', 1)[0]

    def get_raw_signal(self):
        try:
            with h5py.File(self.file, 'r') as data:
                raw_dat = list(data[self.READS_GROUP].values())[0]
                raw_dat = raw_dat['Signal'][()]  # FIXME something missing?
        except Exception:  # FIXME
            raise RuntimeError('Raw data is not stored in Raw/Reads/Read_[read#] so '
                               'new segments cannot be identified.')
        return raw_dat

    def get_events(self, group, subgroup):
        try:
            with h5py.File(self.file, 'r') as data:
                event = data[f'/Analyses/{group}/{subgroup}/Events']
                corr_attrs = dict(list(event.attrs.items()))
                read_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']

                starts = list(map(lambda x: x + read_start_rel_to_raw, event['start']))
                lengths = event['length'].astype(np.int)
                base = [x.decode("UTF-8") for x in event['base']]

        except Exception:  # FIXME
            raise RuntimeError('Events not found.')

        assert len(starts) == len(lengths) == len(base)
        return list(zip(starts, lengths, base))

    def get_alignment_info(self, group, subgroup):
        strand, alignstrand, chrom, chrom_start = [''] * 4

        strand_path = '/'.join(['Analyses', group, subgroup, 'Alignment'])
        try:
            with h5py.File(self.file, 'r') as data:
                alignment_attrs = data[strand_path].attrs

                strand = 't' if subgroup.endswith('template') else 'c'
                if sys.version_info[0] >= 3:  # FIXME what for?
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

        except IOError:
            logging.error("the {} can't be opened".format(self.file))
        except KeyError:
            pass

        return strand, alignstrand, chrom, chrom_start
