#!/usr/bin/envs python3
import os
import click
import shutil
import random
from tqdm import tqdm



@click.command(short_help='mix complement and template from ecoli')
@click.option(
    '-td', '--template_dir', required=True, help='directory to template ecoli'
)
@click.option(
    '-cd', '--complement_dir', required=True, help='dir to complement ecoli'
)
@click.option(
    '-p', '--percentage', default=20, help='kmer length'
)
@click.option(
    '-o', '--output', help='output path'
)
def main(template_dir, complement_dir, percentage, output):
    temp = os.listdir(template_dir)
    files_temp = random.sample(temp, round(len(temp) * percentage / 100))

    comp = os.listdir(complement_dir)
    files_comp = random.sample(comp, 295)

    for el in tqdm(files_temp):
        in_file = os.path.join(template_dir, el)
        out_file = os.path.join(output, '{}_template'.format(el))
        shutil.move(in_file, out_file)

    for el in tqdm(files_comp):
        in_file = os.path.join(complement_dir, el)
        out_file = os.path.join(output, '{}_complement'.format(el))
        shutil.move(in_file, out_file)


if __name__ == "__main__":
    main()