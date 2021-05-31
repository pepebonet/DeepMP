#!/usr/bin/envs python3

import os
import click 
import numpy as np
import pandas as pd 

from tqdm import tqdm


def _write_to_file(file, content, attach=False):
    if attach and os.path.exists(file):
        open_flag = 'a'
    else:
        open_flag = 'w'

    with open(file, open_flag) as f:
        f.write(str(content))


def load_txt(path):
    chrom, read_id, loc_in_genome_start, loc_in_genome_end, prob_cs = [], [], [], [], []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            ch, rid, gen_s, gen_e, pcs = arrange_line(line.split('\t'))
            chrom.append(ch); read_id.append(rid); prob_cs.append(pcs)
            loc_in_genome_start.append(gen_s); loc_in_genome_end.append(gen_e)

    chrom_all = np.concatenate(chrom)        
    read_id_all = np.concatenate(read_id)        
    prob_cs_all = np.concatenate(prob_cs)        
    loc_in_genome_start_all = np.concatenate(loc_in_genome_start)        
    loc_in_genome_end_all = np.concatenate(loc_in_genome_end)        
    import pdb;pdb.set_trace()
    return pd.DataFrame(list(zip(chrom_all, read_id_all, \
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
        except:
            pos_cs = [i for i, ltr in enumerate(line[4]) if ltr == 'G']
            loc_in_read_start.append(pos_cs[j + sum_pre])
            loc_in_read_end.append(pos_cs[j + sum_pre] + 1)
            loc_in_genome_start.append(start_read + pos_cs[j + sum_pre])
            loc_in_genome_end.append(start_read + pos_cs[j + sum_pre] + 1)
            sum_pre += j + 1

    return chrom, read_id, loc_in_genome_start, loc_in_genome_end, prob_cs
    
    
def arrange_results(data):
    import pdb;pdb.set_trace()


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-mp', '--megalodon_path', required=True, 
    help='Output table from megalodon from sam'
)
@click.option(
    '-o', '--output', default='', help='output path'
)
def main(megalodon_path, output):
    megalodon = load_txt(megalodon_path)

    df_megalodon = arrange_results(megalodon)
    import pdb;pdb.set_trace()



if __name__ == '__main__':
    main()