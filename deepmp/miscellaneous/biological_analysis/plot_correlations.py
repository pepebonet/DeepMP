

import click




# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-do', '--deepmp_output', default='', 
    help='Output table from deepMP'
)
@click.option(
    '-dso', '--deepsignal_output', default='', 
    help='Output table from deepsignal'
)
@click.option(
    '-no', '--nanopolish_output', default='', 
    help='nanopolish output table'
)
@click.option(
    '-go', '--guppy_output', default='', 
    help='guppy output table'
)
@click.option(
    '-mo', '--megalodon_output', default='', 
    help='megalodon output table'
)
@click.option(
    '-bp', '--bisulfite_positions', default='', 
    help='posiitions and methylation frequency given by bisulfite sequencing'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(deepmp_output, deepsignal_output, nanopolish_output, guppy_output, 
    megalodon_output, original_test, output):
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
