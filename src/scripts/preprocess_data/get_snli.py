#!/usr/bin/env python

'''
Downloads SNLI dataset as zip file and unpacks it.
'''

import os

from src import DATA_DIR
from src.util.get_data import get_data

if __name__ == '__main__':
    get_data('https://nlp.stanford.edu/projects/snli/snli_1.0.zip', 'raw')

    if not os.path.exists(os.path.join(DATA_DIR, 'snli')):
        os.makedirs(os.path.join(DATA_DIR, 'snli'))
