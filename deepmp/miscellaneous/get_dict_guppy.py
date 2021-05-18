#!/usr/bin/env python3 

import click
import pickle
import pandas as pd
from collections import defaultdict

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@click.command(short_help='Get dict of read names')
@click.option('-dt', '--data_type', type=click.Choice(['Ecoli', 'Human']), required=True)
@click.option('-ss', '--sequencing_summary', required=True, help='fastq file')
@click.option('-o', '--output', required=True,help='Path to save dict')
def main(data_type, sequencing_summary, output):
    read_names = defaultdict()
    summary = pd.read_csv(sequencing_summary, sep='\t')
    summary['read_id_txt'] = summary['read_id'] + '.txt'
    filename = summary['filename'].tolist()
    read_id = summary['read_id_txt'].tolist()

    if data_type == 'Ecoli':
        for x, y in zip(filename, read_id):
            read_names[x] = y
            
    else:
        import pdb;pdb.set_trace()
        for x, y in zip(filename, read_id):
            read_names[x] = y

    save_obj(output, read_names)


if __name__ == "__main__":
    main()