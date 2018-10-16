# -*- coding: utf-8 -*-
"""
 Fetchers for publicly available pretrained embeddings,
 adapted from https://github.com/kudkudak/word-embeddings-benchmarks and polyglot package

 TODO: This is not really needed, we represent embeddings as simple h5 arrays, refactor
"""

import pickle as pickle
import logging
from functools import partial

import numpy as np
from six import PY2
from six import text_type
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

from io import open
from collections import Counter

import tqdm
import h5py
import six
from os import path
import io
import re

from six import iteritems
from six import text_type as str
from six import string_types


def _open(file_, mode='r'):
    """Open file object given filenames, open files or even archives."""
    if isinstance(file_, string_types):
        _, ext = path.splitext(file_)
        if ext in {'.gz'}:
            if mode == "r" or mode == "rb":
                # gzip is extremely slow
                return io.BufferedReader(gzip.GzipFile(file_, mode=mode))
            else:
                return gzip.GzipFile(file_, mode=mode)
        if ext in {'.bz2'}:
            return bz2.BZ2File(file_, mode=mode)
        else:
            return io.open(file_, mode)
    return file_


def _remove_chars(string, old='/', new='%', meta='_'):
    assert isinstance(old, str)
    assert isinstance(new, str)
    assert isinstance(meta, str)
    assert len(set([old, new, meta])) == 3
    assert list(map(len, [old, new, meta])) == [1, 1, 1]
    return re.sub(re.escape(old), meta + new, re.sub(re.escape(meta), meta + meta, string))


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, text_type):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return text_type(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8

# Works just as good with unicode chars
_delchars = [chr(c) for c in range(256)]
_delchars = [x for x in _delchars if not x.isalnum()]
_delchars.remove('\t')
_delchars.remove(' ')
_delchars.remove('-')
_delchars.remove('_')  # for instance phrases are joined in word2vec used this char
_delchars = ''.join(_delchars)
_delchars_table = dict((ord(char), None) for char in _delchars)


def standardize_string(s, clean_words=True, lower=True, language="english"):
    """
    Ensures common convention across code. Converts to utf-8 and removes non-alphanumeric characters
    Parameters
    ----------
    language: only "english" is now supported. If "english" will remove non-alphanumeric characters
    lower: if True will lower strńing.
    clean_words: if True will remove non alphanumeric characters (for instance '$', '#' or 'ł')
    Returns
    -------
    string: processed string
    """

    assert isinstance(s, string_types)

    if not isinstance(s, text_type):
        s = text_type(s, "utf-8")

    if language == "english":
        s = (s.lower() if lower else s)
        s = (s.translate(_delchars_table) if clean_words else s)
        return s
    else:
        raise NotImplementedError("Not implemented standarization for other languages")


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


def load_embedding(f_name):
    logging.info("Loading " + f_name)
    f = h5py.File(f_name)
    words = f['words_flatten'][0].split("\n")
    vectors = f['embedding'][:]
    return dict(list(zip(words, vectors)))


def count(lines):
    """ Counts the word frequences in a list of sentences.

    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    """
    words = [w for l in lines for w in l.strip().split()]
    return Counter(words)


class Vocabulary(object):
    """ A set of words/tokens that have consistent IDs.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list/set): list or set of words.
        """
        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def __iter__(self):
        """Iterate over the words in a vocabulary."""
        for w, i in sorted(iteritems(self.word_id), key=lambda wc: wc[1]):
            yield w

    @property
    def words(self):
        """ Ordered list of words according to their IDs."""
        return list(self)

    def __unicode__(self):
        return "\n".join(self.words)

    def __str__(self):
        if six.PY3:
            return self.__unicode__()
        return self.__unicode__().encode("utf-8")

    def __getitem__(self, key):
        if isinstance(key, string_types) and not isinstance(key, str):
            key = str(key, encoding="utf-8")
        return self.word_id[key]

    def add(self, word):
        if isinstance(word, string_types) and not isinstance(word, str):
            word = str(word, encoding="utf-8")

        if word in self.word_id:
            raise RuntimeError("Already existing word")

        id = len(self.word_id)
        self.word_id[word] = id
        self.id_word[id] = word

    def update(self, D):
        raise NotImplementedError()

    def __contains__(self, key):
        return key in self.word_id

    def __delitem__(self, key):
        """Delete a word from vocabulary.

        Note:
         To maintain consecutive IDs, this operation implemented
         with a complexity of \\theta(n).
        """
        del self.word_id[key]
        self.id_word = dict(enumerate(self.words))
        self.word_id = {w: i for i, w in iteritems(self.id_word)}

    def __len__(self):
        return len(self.word_id)

    def sanitize_words(self, words):
        """Guarantees that all textual symbols are unicode.
        Note:
          We do not convert numbers, only strings to unicode.
          We assume that the strings are encoded in utf-8.
        """
        _words = []
        for w in words:
            if isinstance(w, string_types) and not isinstance(w, str):
                _words.append(str(w, encoding="utf-8"))
            else:
                _words.append(w)
        return _words

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def getstate(self):
        return list(self.words)

    @classmethod
    def from_vocabfile(cls, filename):
        """ Construct a CountedVocabulary out of a vocabulary file.

        Note:
          File has the following format word1
                                        word2
        """
        words = [x.strip() for x in _open(filename, 'r').read().splitlines()]
        return cls(words=words)


class OrderedVocabulary(Vocabulary):
    """ An ordered list of words/tokens according to their frequency.

    Note:
      The words order is assumed to be sorted according to the word frequency.
      Most frequent words appear first in the list.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list): list of sorted words according to frequency.
        """

        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        return OrderedVocabulary(words=self.words[:k])


class CountedVocabulary(OrderedVocabulary):
    """ List of words and counts sorted according to word count.
    """

    def __init__(self, word_count=None):
        """ Build attributes word_id and id_word from input.

        Args:
          word_count (dictionary): A dictionary of the type word:count or
                                   list of tuples of the type (word, count).
        """

        if isinstance(word_count, dict):
            word_count = iteritems(word_count)
        sorted_counts = list(sorted(word_count, key=lambda wc: wc[1], reverse=True))
        words = [w for w, c in sorted_counts]
        super(CountedVocabulary, self).__init__(words=words)
        self.word_count = dict(sorted_counts)

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        word_count = {w: self.word_count[w] for w in self.words[:k]}
        return CountedVocabulary(word_count=word_count)

    def min_count(self, n=1):
        """ Returns a vocabulary after eliminating the words that appear < `n`.

        Args:
          n (integer): specifies the minimum word frequency allowed.
        """
        word_count = {w: c for w, c in iteritems(self.word_count) if c >= n}
        return CountedVocabulary(word_count=word_count)

    def __unicode__(self):
        return "\n".join(["{}\t{}".format(w, self.word_count[w]) for w in self.words])

    def __delitem__(self, key):
        super(CountedVocabulary, self).__delitem__(key)
        self.word_count = {w: self.word_count[w] for w in self}

    def getstate(self):
        words = list(self.words)
        counts = [self.word_count[w] for w in words]
        return (words, counts)

    @staticmethod
    def from_vocabfile(filename):
        """ Construct a CountedVocabulary out of a vocabulary file.

        Note:
          File has the following format word1 count1
                                        word2 count2
        """
        word_count = [x.strip().split() for x in _open(filename, 'r').read().splitlines()]
        word_count = {w: int(c) for w, c in word_count}
        return CountedVocabulary(word_count=word_count)


class Embedding(object):
    """ Mapping a vocabulary to a d-dimensional points."""

    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = np.asarray(vectors)
        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors."
                .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):
            logger.warning("Vocabulary has duplicates.")

    def __getitem__(self, k):
        return self.vectors[self.vocabulary[k]]

    def __setitem__(self, k, v):
        if not v.shape[0] == self.vectors.shape[1]:
            raise RuntimeError("Please pass vector of len {}".format(self.vectors.shape[1]))

        if k not in self.vocabulary:
            self.vocabulary.add(k)
            self.vectors = np.vstack([self.vectors, v.reshape(1, -1)])
        else:
            self.vectors[self.vocabulary[k]] = v

    def __contains__(self, k):
        return k in self.vocabulary

    def __delitem__(self, k):
        """Remove the word and its vector from the embedding.

        Note:
         This operation costs \\theta(n). Be careful putting it in a loop.
        """
        index = self.vocabulary[k]
        del self.vocabulary[k]
        self.vectors = np.delete(self.vectors, index, 0)

    def __len__(self):
        return len(self.vocabulary)

    def __iter__(self):
        for w in self.vocabulary:
            yield w, self[w]

    @property
    def words(self):
        return self.vocabulary.words

    @property
    def shape(self):
        return self.vectors.shape

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def standardize_words(self, lower=False, clean_words=False, inplace=False):
        return self.transform_words(partial(standardize_string, lower=lower, clean_words=clean_words),
            inplace=inplace)

    def transform_words(self, f, inplace=False):
        """ Transform words in vocabulary """
        id_map = {}
        id_map_to_new = {}
        word_count = len(self.vectors)
        if inplace:
            for id, w in enumerate(self.vocabulary.words):
                fw = f(w)
                if len(fw) and (fw not in id_map):
                    id_map[fw] = id
                    id_map_to_new[fw] = len(id_map) - 1
                    self.vectors[len(id_map) - 1] = self.vectors[id]
                elif len(fw) and (fw in id_map) and (fw == w):
                    # Overwrites with last occurrence
                    self.vectors[id_map_to_new[fw]] = self.vectors[id]

            words = sorted(list(id_map.keys()), key=lambda x: id_map[x])
            self.vectors = self.vectors[0:len(id_map)]
            self.vocabulary = self.vocabulary.__class__(words)
            logger.info("Transformed {} into {} words".format(word_count, len(words)))
            return self
        else:
            for id, w in enumerate(self.vocabulary.words):
                if len(f(w)) and (f(w) not in id_map or f(w) == w):
                    id_map[f(w)] = id
            words = sorted(list(id_map.keys()), key=lambda x: id_map[x])
            vectors = self.vectors[[id_map[w] for w in words]]
            logger.info("Transformed {} into {} words".format(word_count, len(words)))
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """Only most frequent k words to be included in the embeddings."""

        assert isinstance(self.vocabulary, OrderedVocabulary), \
            "most_frequent can be called only on Embedding with OrderedVocabulary"

        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """Normalize embeddings matrix row-wise.

        Parameters
        ----------
          ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:
            ord = None  # numpy uses this flag to indicate l2.
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
        Find nearest neighbor of given word

        Parameters
        ----------
          word: string or vector
            Query word or vector.

          k: int, default: 1
            Number of nearest neihbours to return.

          metric: string, default: 'cosine'
            Metric to use.

          exclude: list, default: []
            Words to omit in answer

        Returns
        -------
          n: list
            Nearest neighbors.
        """
        if isinstance(word, string_types):
            assert word in self, "Word not found in the vocabulary"
            v = self[word]
        else:
            v = word

        D = pairwise_distances(self.vectors, v.reshape(1, -1), metric=metric)

        if isinstance(word, string_types):
            D[self.vocabulary.word_id[word]] = D.max()

        for w in exclude:
            D[self.vocabulary.word_id[w]] = D.max()

        return [self.vocabulary.id_word[id] for id in D.argsort(axis=0).flatten()[0:k]]

    @staticmethod
    def from_gensim(model):
        word_count = {}
        vectors = []
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            word = standardize_string(word)
            if word:
                vectors.append(model.syn0[vocab.index])
                word_count[word] = vocab.count
        vocab = CountedVocabulary(word_count=word_count)
        vectors = np.asarray(vectors)
        return Embedding(vocabulary=vocab, vectors=vectors)

    @staticmethod
    def from_word2vec_vocab(fvocab):
        counts = {}
        with _open(fvocab) as fin:
            for line in fin:
                word, count = standardize_string(line).strip().split()
                if word:
                    counts[word] = int(count)
        return CountedVocabulary(word_count=counts)

    @staticmethod
    def _from_word2vec_binary(fname):
        with _open(fname, 'rb') as fin:
            words = []
            header = fin.readline()
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            logger.info("Loading #{} words with {} dim".format(vocab_size, layer1_size))
            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)
            binary_len = np.dtype("float32").itemsize * layer1_size
            for line_no in tqdm.tqdm(list(range(vocab_size)), total=vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        word.append(ch)

                words.append(b''.join(word).decode("latin-1"))
                vectors[line_no, :] = np.fromstring(fin.read(binary_len), dtype=np.float32)

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname):
        with _open(fname, 'r') as fin:
            words = []
            header = fin.readline()
            ignored = 0
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)
            for line_no, line in tqdm.tqdm(enumerate(fin), total=vocab_size):
                try:
                    parts = text_type(line, encoding="utf-8").strip().split()
                except TypeError as e:
                    parts = line.strip().split()
                except Exception as e:
                    logger.warning("We ignored line number {} because of erros in parsing"
                                   "\n{}".format(line_no, e))
                    continue
                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.
                if len(parts) == layer1_size + 1:
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:]))
                elif len(parts) == layer1_size + 2:
                    word, vectors[line_no - ignored] = parts[:2], list(map(np.float32, parts[2:]))
                    word = " ".join(word)
                else:
                    ignored += 1
                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))
                    continue

                words.append(word)
            if ignored:
                vectors = vectors[0:-ignored]

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def from_glove(fname, vocab_size, dim):
        with _open(fname, 'r') as fin:
            words = []
            ignored = 0
            vectors = np.zeros(shape=(vocab_size, dim), dtype=np.float32)
            for line_no, line in tqdm.tqdm(enumerate(fin), total=vocab_size):
                line = line.strip()

                if "\ufffd" in line:  # hack to remove particular non-unicode duplicate after decoding
                    ignored += 1
                    logger.warning("We ignored line number {} because of a bad character".format(line_no))
                    continue

                try:
                    parts = text_type(line, encoding="utf-8").split(" ")  # not the same as .split()
                except TypeError as e:
                    parts = line.split(" ")  # not the same as .split()
                except Exception as e:
                    ignored += 1
                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))
                    continue

                try:
                    word, vectors[line_no - ignored] = " ".join(parts[0:len(parts) - dim]), list(
                        map(np.float32, parts[len(parts) - dim:]))
                    words.append(word)
                except Exception as e:
                    ignored += 1
                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))
            return Embedding(vocabulary=OrderedVocabulary(words), vectors=vectors[0:len(words)])

    @staticmethod
    def from_dict(d):
        for k in d:  # Standardize
            d[k] = np.array(d[k]).flatten()
        return Embedding(vectors=list(d.values()), vocabulary=Vocabulary(list(d.keys())))

    def to_dict(self):
        d = {}
        for w in self.vocabulary:
            d[w] = self[w]
        return d

    @staticmethod
    def to_word2vec(w, fname, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        w: Embedding instance

        fname: string
          Destination file
        """
        logger.info("storing %sx%s projection weights into %s" % (w.vectors.shape[0], w.vectors.shape[1], fname))
        with _open(fname, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % w.vectors.shape))
            # store in sorted order: most frequent words at the top
            for word, vector in zip(w.vocabulary.words, w.vectors):
                if binary:
                    fout.write(to_utf8(word) + b" " + vector.astype("float32").tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%.15f" % val for val in vector))))

    @staticmethod
    def from_word2vec(fname, fvocab=None, binary=False):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        vocabulary = None
        if fvocab is not None:
            logger.info("loading word counts from %s" % (fvocab))
            vocabulary = Embedding.from_word2vec_vocab(fvocab)

        logger.info("loading projection weights from %s" % (fname))
        if binary:
            words, vectors = Embedding._from_word2vec_binary(fname)
        else:
            words, vectors = Embedding._from_word2vec_text(fname)

        if not vocabulary:
            vocabulary = OrderedVocabulary(words=words)

        if len(words) != len(set(words)):
            raise RuntimeError("Vocabulary has duplicates")

        e = Embedding(vocabulary=vocabulary, vectors=vectors)

        return e

    @staticmethod
    def load(fname):
        """Load an embedding dump generated by `save`"""

        content = _open(fname).read()
        if PY2:
            state = pickle.loads(content)
        else:
            state = pickle.loads(content, encoding='latin1')
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(list(zip(words, counts)))
            vocab = CountedVocabulary(word_count=word_count)
        else:
            vocab = OrderedVocabulary(voc)
        return Embedding(vocabulary=vocab, vectors=vec)

    def save(self, fname):
        """Save a pickled version of the embedding into `fname`."""

        vec = self.vectors
        voc = self.vocabulary.getstate()
        state = (voc, vec)
        with open(fname, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
