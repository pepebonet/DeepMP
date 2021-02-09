#!/usr/bin/env pyhton3
import ast
import click
import pickle
from Bio import SeqIO
from collections import defaultdict

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@click.command(short_help='Get dict of read names')
@click.option('-dt', '--data_type', type=click.Choice(['Ecoli', 'Human']), required=True)
@click.option('-f', '--fastq', required=True, help='fastq file')
@click.option('-o', '--output', required=True,help='Path to save dict')
def main(data_type, fastq, output):
    read_names = defaultdict()
    for record in SeqIO.parse(fastq, "fastq"):
        
        if data_type == 'Ecoli':
            names = record.description.split(' ')
            read_names[names[0]] = names[1]

        else:
            try:
                names = record.description.split(' ', 1)
                read_name_human = ast.literal_eval(names[1][1:])['filename']
                read_names[names[0]] = read_name_human
            except:
                names = record.description.split(' ')
                read_name_human = names[5].split(',')[0].replace('"', '')
                read_names[names[0]] = read_name_human

        
    save_obj(output, read_names)


if __name__ == "__main__":
    main()