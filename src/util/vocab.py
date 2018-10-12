# -*- coding: utf-8 -*-



from collections import Counter
import logging
from six import string_types, text_type
import numpy
import time
import numpy as np
logger = logging.getLogger()


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
        #s = (s.lower() if lower else s)
        #s = (s.translate(_delchars_table) if clean_words else s)
        return s
    else:
        raise NotImplementedError("Not implemented standarization for other languages")


class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""
    BOS = b'<bos>'  # beginning-of-sequence
    EOS = b'<eos>'  # end-of-sequence
    BOD = b'<bod>'  # beginning-of-definition
    EOD = b'<eod>'  # end-of-definition
    UNK = b'<unk>'  # unknown token
    PAD = b'<pad>'  # padding token
    SPECIAL_TOKEN_MAP = {
        BOS: 'bos',
        EOS: 'eos',
        BOD: 'bod',
        EOD: 'eod',
        UNK: 'unk',
        PAD: 'pad',
    }

    def __init__(self, path_or_data):
        """Initialize the vocabulary.
        path_or_data
            Either a list of words or the path to it.
        top_k
            If not `None`, only the first `top_k` entries will be left.
            Note, this does not include the special tokens.
        """
        if isinstance(path_or_data, str):
            words_and_freqs = []
            with open(path_or_data, "rb") as f:
                for line in f:
                    word, freq_str = line.strip().split()
                    freq = int(freq_str)
                    words_and_freqs.append((word, freq))
        else:
            words_and_freqs = path_or_data

        self._id_to_word = []
        self._id_to_freq = []
        self._word_to_id = {}
        self.bos = self.eos = -1
        self.bod = self.eod = -1
        self.unk = -1
        self.pad = -1

        for idx, (word_name, freq) in enumerate(words_and_freqs):
            token_attr = self.SPECIAL_TOKEN_MAP.get(word_name)
            if token_attr is not None:
                setattr(self, token_attr, idx)
            self._id_to_word.append(word_name)
            self._id_to_freq.append(freq)
            self._word_to_id[word_name] = idx

        if -1 in [getattr(self, attr)
                  for attr in list(self.SPECIAL_TOKEN_MAP.values())]:
            pass
            # raise ValueError("special token not found in the vocabulary")

        logger.info("Vocab loaded from:", path_or_data)
        for attr in list(self.SPECIAL_TOKEN_MAP.values()):
            print(attr, getattr(self, attr))

    def size(self):
        return len(self._id_to_word)

    @property
    def words(self):
        return self._id_to_word

    @property
    def frequencies(self):
        return self._id_to_freq

    def word_to_id(self, word, top_k=None):
        id_ = self._word_to_id.get(word)
        if id_ is not None:
            if not top_k or id_ < top_k:
                return id_
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        _cur_ids = cur_ids[np.nonzero(cur_ids)]
        try:
            return ' '.join([self.id_to_word(cur_id) for cur_id in _cur_ids])
        except:
            return b' '.join([self.id_to_word(cur_id) for cur_id in _cur_ids])

    def encode(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence]
        return numpy.array(word_ids, dtype=numpy.int64)

    @staticmethod
    def build(filename_or_words, top_k=None, sort_by='frequency'):
        """
        sort_by is either 'frequency' or 'lexicographical'
        """
        # For now let's use a very stupid tokenization
        if isinstance(filename_or_words, str):
            with open(filename_or_words) as file_:
                def data():
                    for line in file_:
                        for word in line.strip().split():
                            yield word
                counter = Counter(data())
            logger.info("Data is read")
        else:
            counter = Counter(filename_or_words)
            for word in list(counter.keys()):
                if ' ' in word:
                    logger.error("can't have tokens with spaces, skip {}".format(word))
                    del counter[word]
        # It was not immediately clear to me
        # if counter.most_common() selects consistenly among
        # the words with the same counts. Hence, let's just sort.
        if sort_by == 'frequency':
            sortf = lambda x: (-x[1], x[0])
        elif sort_by == 'lexicographical':
            sortf = lambda x: (x[0], x[1])
        else:
            raise Exception("sort not understood:", sort_by)
        words_and_freqs = sorted(list(counter.items()), key=sortf)
        logger.info("Words are sorted")
        if top_k:
            words_and_freqs = words_and_freqs[:top_k]
        words_and_freqs = (
            [(Vocabulary.PAD, 0),
             (Vocabulary.UNK, 0),
             (Vocabulary.EOS, 0),
             (Vocabulary.BOD, 0),
             (Vocabulary.EOD, 0),
             (Vocabulary.BOS, 0)]
            + words_and_freqs)

        return Vocabulary(words_and_freqs)

    def save(self, filename):
        with open(filename, 'w') as f:
            for word, freq in zip(self._id_to_word, self._id_to_freq):
                word = standardize_string(word)
                print(word, freq, file=f)