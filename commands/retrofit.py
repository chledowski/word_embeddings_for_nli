"""
The `retrofit` command modifies given embeddings with Faruqui method.
"""

import copy
import h5py
import numpy
import re

from scipy.stats import ortho_group
from tqdm import trange

from common.paths import *

# TODO(kchledowski): Refactor this code.


def load_embedding(name):
    path = os.path.join(EMBEDDINGS_DIR, name + ".h5")
    emb_file = h5py.File(path, 'r')
    emb_words = emb_file['words_flatten'][0].split('\n')
    emb_matrix_all = emb_file[list(emb_file.keys())[0]][:]

    wv = {}
    for i in range(len(emb_words)):
        wv[emb_words[i]] = emb_matrix_all[i]

    return emb_words, emb_matrix_all, wv


isNumber = re.compile(r'\d+.*')


def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


def load_lexicon(filepath):
    lexicon = {}
    for line in open(filepath, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def export_embedding_h5(vocabulary, embedding_matrix, output='embedding.h5'):
    f = h5py.File(output, "w")
    compress_option = dict(compression="gzip", compression_opts=9, shuffle=True)
    words_flatten = '\n'.join(vocabulary)
    # print(words_flatten)
    f.attrs['vocab_len'] = len(vocabulary)
    dt = h5py.special_dtype(vlen=str)
    _dset_vocab = f.create_dataset('words_flatten', (1,), dtype=dt, **compress_option)
    # print(_dset_vocab[0])
    _dset_vocab[...] = [words_flatten]
    _dset = f.create_dataset('embedding', embedding_matrix.shape, dtype=embedding_matrix.dtype, **compress_option)
    _dset[...] = embedding_matrix
    f.flush()
    f.close()


def export_dict_to_h5(wv, path):
    retro_vocab = []
    retro_matrix = []

    for key in wv.keys():
        retro_vocab.append(key)
        retro_matrix.append(wv[key])

    export_embedding_h5(retro_vocab, numpy.array(retro_matrix), output=path)


def faruqui(wv_original, lexicon, args):
    if 1 in args.losses:
        a = args.alpha
    else:
        a = 0

    retro_wv = copy.deepcopy(wv_original)
    vocab_wv = set(retro_wv.keys())
    vocab_intersection = vocab_wv.intersection(set(lexicon.keys()))

    for it in trange(args.epochs):
        # loop through every node also in ontology (else just use data estimate)
        for word in vocab_intersection:
            word_neighbours = set(lexicon[word]).intersection(vocab_wv)
            n_neighbours = len(word_neighbours)
            # no neighbours, pass - use data estimate
            if n_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            newVec = a * n_neighbours * wv_original[word]
            # loop over neighbours and add to new vector (currently with weight 1)
            for word_neighbour in word_neighbours:
                newVec += args.beta * retro_wv[word_neighbour]
            retro_wv[word] = newVec / ((1 + args.beta) * n_neighbours)

    return retro_wv, []


def calc_norm_of_wv(wv_before, wv_after, lexicon):
    i = 0
    norm_sum = 0
    lexi_keys = set(lexicon.keys())
    for key in wv_before.keys():
        if key in lexi_keys:
            norm_sum += numpy.linalg.norm(wv_after[key]) / numpy.linalg.norm(wv_before[key])
            i += 1


def retrofit_from_args(args):
    emb_words, emb_matrix_all, wv = load_embedding(args.in_embedding)

    lexicon = load_lexicon(
        os.path.join(LEXICONS_DIR, args.lexicon + '.txt'))

    wv_2, losses = faruqui(wv, lexicon, args)

    if args.sum:
        if args.q:
            Q = ortho_group.rvs(dim=300)
            wv_q = {}
            for key in wv_2.keys():
                wv_q[key] = Q @ wv_2[key]

            wv_2 = wv_q

        for key in wv_2.keys():
            wv_2[key] = wv[key] + wv_2[key]

    export_dict_to_h5(wv_2, os.path.join(EMBEDDINGS_DIR, "%s.h5" % args.out_embedding))


def retrofit_from_parser(parser):
    parser.add_argument("--in_embedding", type=str, required=True)
    parser.add_argument("--out_embedding", type=str, required=True)
    parser.add_argument('--lexicon', default='wordnet-synonyms+', type=str)

    # Faruqui parameters
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument('--losses', nargs='+', default=[1, 2])
    parser.add_argument("--epochs", default=10, type=int)

    # Sum two embeddings?
    parser.add_argument("--sum", action='store_true')

    # Rotate before summation?
    parser.add_argument("--q", action='store_true')

    args = parser.parse_args()
    retrofit_from_args(args)
