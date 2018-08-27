import argparse
import codecs
import h5py
import numpy as np
import os
from tqdm import tqdm

from src import DATA_DIR


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

    export_embedding_h5(retro_vocab, np.array(retro_matrix),
                        output=path)


def h5_to_txt(h5_name, txt_name, prefix="en_"):
    emb_file = h5py.File(os.path.join(DATA_DIR, 'embeddings', h5_name + ".h5"), 'r')
    emb_words = emb_file['words_flatten'][0].split('\n')
    emb_matrix_all = emb_file[list(emb_file.keys())[0]][:]
    # emb_matrix_all = normalize_embeddings(emb_matrix_all)

    txt_path = os.path.join(DATA_DIR, 'embeddings', txt_name + ".txt")

    if not os.path.exists(txt_path):
        with open(txt_path, 'w') as text_file:
            for i in tqdm(range(len(emb_words))):
                word = emb_words[i]
                str_embedding = ' '.join(list(map(str, emb_matrix_all[i])))
                text_file.write("%s%s %s\n" % (prefix, word, str_embedding))


def txt_to_h5(h5_name, txt_name):
    word_dictionary = {}

    f = codecs.open(os.path.join(DATA_DIR, 'embeddings', txt_name + ".txt"), 'r')

    for line in f:
        try:
            line = line.split(" ", 1)
            key = line[0].lower()
            vect = np.fromstring(line[1], dtype="float32", sep=" ")
            word_dictionary[key[3:]] = vect / np.linalg.norm(vect)
        except:
            print(line)
    export_dict_to_h5(word_dictionary, os.path.join(DATA_DIR, "embeddings", h5_name + ".h5"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default='LEAR_ONLY.h5', type=str)
    parser.add_argument("--txt", default='wv_final', type=str)
    parser.add_argument("--prefix", default='en_', type=str)
    parser.add_argument("--convert-to", default="h5", type=str, help='choose from txt and h5')
    args = parser.parse_args()
    if args.convert_to == 'txt':
        h5_to_txt(args.h5, args.txt, args.prefix)
    elif args.convert_to == 'h5':
        txt_to_h5(args.h5, args.txt)
