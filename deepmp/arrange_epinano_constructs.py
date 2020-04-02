#!/usr/bin/env python3
import os
import click
import pandas as pd





# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.argument(
    'input'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(input, output):
    data = pd.read_csv(input, sep=',')
    data = data[data['std_current'].notna()]
    #TODO from this data obtain errors and sequence features 
    import pdb;pdb.set_trace()
    
    
if __name__ == "__main__":
    main()