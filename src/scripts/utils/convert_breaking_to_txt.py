import argparse
import h5py
import os

from fuel.datasets.hdf5 import H5PYDataset
from src import DATA_DIR


def main(h5_filename):
    dic_label = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    h5_file = h5py.File(os.path.join(DATA_DIR, 'snli', 'test_breaking_nli.h5'), 'r')

    txt_filename = ''.join(h5_filename.split('.')[:-1]) + '.txt'
    with open(txt_filename, 'w') as text_file:
        text_file.write("label\tsentence1\tsentence2\n")
        for label, s1, s2 in zip(h5_file['label'], h5_file['sentence1'], h5_file['sentence2']):
            s1 = ' '.join([word.decode('utf-8') for word in s1])
            s2 = ' '.join([word.decode('utf-8') for word in s2])
            text_file.write("%s\t%s\t%s\n" % (dic_label[label], s1, s2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-filename', type=str)
    args = parser.parse_args()

    main(args.h5_filename)