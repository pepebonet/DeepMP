#!/usr/bin/envs python3

import os
import sys
import click 
import numpy as np
import pandas as pd 

from tqdm import tqdm

sys.path.append('../')
import deepmp.utils as ut

def load_txt(path):
    chrom, read_id, loc_in_genome_start, loc_in_genome_end, \
        prob_cs, strands = [], [], [], [], [], []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            ch, rid, gen_s, gen_e, pcs, strand = arrange_line(line.split('\t'))
            chrom.append(ch); read_id.append(rid); prob_cs.append(pcs)
            loc_in_genome_start.append(gen_s); loc_in_genome_end.append(gen_e)
            strands.append(strand)

    chrom_all = np.concatenate(chrom)        
    strand_all = np.concatenate(strands)        
    read_id_all = np.concatenate(read_id)        
    prob_cs_all = np.concatenate(prob_cs)        
    loc_in_genome_start_all = np.concatenate(loc_in_genome_start)        
    loc_in_genome_end_all = np.concatenate(loc_in_genome_end)        

    return pd.DataFrame(list(zip(chrom_all, read_id_all, strand_all, \
        loc_in_genome_start_all, loc_in_genome_end_all, prob_cs_all)))


def arrange_line(line):
    meth_cs = [np.int(el) for el in line[5][:-1].split(',')[1:]]
    prob_cs = [np.int(el) / 255 for el in line[6][:-1].split(',')[1:]]
    pos_cs = [i for i, ltr in enumerate(line[4]) if ltr == 'C']
    read_id = [line[0]] * len(prob_cs)
    chrom = [line[1]] * len(prob_cs)
    strand = []
    start_read = np.int(line[2])
    loc_in_genome_start, loc_in_genome_end = [], []
    loc_in_read_start, loc_in_read_end = [], []
    sum_pre = 0
    for i, j in enumerate(meth_cs):
        try: 
            loc_in_read_start.append(pos_cs[j + sum_pre])
            loc_in_read_end.append(pos_cs[j + sum_pre] + 1)
            loc_in_genome_start.append(start_read + pos_cs[j + sum_pre])
            loc_in_genome_end.append(start_read + pos_cs[j + sum_pre] + 1)
            sum_pre += j + 1
            strand.append('+')
        except:
            pos_cs = [i for i, ltr in enumerate(line[4]) if ltr == 'G']
            loc_in_read_start.append(pos_cs[j + sum_pre])
            loc_in_read_end.append(pos_cs[j + sum_pre] + 1)
            loc_in_genome_start.append(start_read + pos_cs[j + sum_pre])
            loc_in_genome_end.append(start_read + pos_cs[j + sum_pre] + 1)
            sum_pre += j + 1
            strand.append('-')

    return chrom, read_id, loc_in_genome_start, loc_in_genome_end, prob_cs, strand


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Script to obtain megalodon output')
@click.option(
    '-mp', '--megalodon_path', required=True, 
    help='Output table from megalodon from sam'
)
@click.option(
    '-dr', '--dict_reads', required=True, 
    help='dict reads'
)
@click.option(
    '-o', '--output', default='', help='output path'
)
def main(megalodon_path, dict_reads, output):
    megalodon = load_txt(megalodon_path)

    if dict_reads: 
        dict_names = ut.load_obj(dict_reads)  
        dict_names = {v: k for k, v in dict_names.items()}
        aa = megalodon[1] + '.txt'
        bb = [dict_names[x] for x in aa.tolist()]
        import pdb;pdb.set_trace()
        megalodon[6] = bb
    megalodon.to_csv(
        os.path.join(output, 'megalodon_results.tsv'), sep='\t', 
        index=None, header=None
    )
    import pdb;pdb.set_trace()



if __name__ == '__main__':
    main()