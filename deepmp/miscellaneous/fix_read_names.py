#!/usr/bin/env pyhton3

import click
import pickle
from Bio import SeqIO
from collections import defaultdict

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@click.command(short_help='Get dict of read names')
@click.option('-f', '--fastq', required=True, help='fastq file')
@click.option('-o', '--output', required=True,help='Path to save dict')
def main(fastq, output):
    read_names = defaultdict()
    for record in SeqIO.parse(fastq, "fastq"):
        names = record.description.split(' ')
        read_names[names[0]] = names[1]
    
    save_obj(output, read_names)


if __name__ == "__main__":
    main()