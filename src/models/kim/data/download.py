"""
Downloads the following:
- Glove vectors
- Stanford Natural Language Inference (SNLI) Corpus
- WordNet 
- CoreNLP tools
"""

import sys
import os
import zipfile
import gzip
import tarfile

from src import DATA_DIR

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def ungzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with tarfile.open(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_snli(dirpath):
    if os.path.exists(dirpath):
        print('Found SNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    unzip(download(url, dirpath))

def download_wordnet(dirpath):
    if os.path.exists(dirpath):
        print('Found WordNet 3.0 - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://wordnetcode.princeton.edu/3.0/WNprolog-3.0.tar.gz'
    ungzip(download(url, dirpath))

def download_corenlp(dirpath, force=True):
    if os.path.exists(dirpath):
        if not force:
            print('Found Stanford CoreNLP - skip')
            return
    else:
        os.makedirs(dirpath)
    url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip'
    unzip(download(url, dirpath))

if __name__ == '__main__':
    snli_dir = os.path.join(DATA_DIR, 'snli')
    wordvec_dir = os.path.join(DATA_DIR, 'glove')
    wordnet_dir = os.path.join(DATA_DIR, 'wordnet')
    corenlp_dir = os.path.join(DATA_DIR, 'corenlp')
    # download_snli(snli_dir)
    # download_wordvecs(wordvec_dir)
    # download_wordnet(wordnet_dir)
    download_corenlp(corenlp_dir)

