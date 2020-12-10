
import sys
import click
import autosklearn.classification

sys.path.append('../')
import deepmp.utils as ut

@click.command(short_help='Trainining autosklearn')
@click.option(
    '-tf', '--train_file', default='',
    help='path to training set'
)
@click.option(
    '-vf', '--val_file', default='',
    help='path to validation set'
)
@click.option(
    '-k', '--kmer', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-md', '--model_dir', default='models/',
    help='directory to trained model'
)
def main(train_file, val_file, kmer, model_dir):
    input_train_seq, input_train_err, label = ut.get_data_jm(train_file, kmer)
    input_val_seq, input_val_err, vy = ut.get_data_jm(val_file, kmer)
    import pdb;pdb.set_trace()
    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)

    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()