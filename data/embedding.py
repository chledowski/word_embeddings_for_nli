#!/usr/bin/env python3

import logging

import h5py
from tqdm import trange

from common.paths import *
from utils.algebra import norm_weight

logger = logging.getLogger(__name__)


class NLIEmbedding(object):
    def __init__(self, rng, vocab, dim, trainable=False, file=None):
        self._dim = dim
        self._vocab = vocab
        self._file = file
        self._rng = rng
        self._trainable = trainable
        self._cached = None

    def load(self, force_reload=False):
        """
        Generate random embeddings or loads embedding from file.
        :param force_reload: if ``True`` it returns cached embedding when possible.
        :return: ``np.ndarray`` embedding matrix for all words from vocabulary.
        """
        if not force_reload and self._cached is not None:
            return self._cached

        target_matrix = norm_weight(self._rng, self._vocab.size(), self.dim)

        if self._file:
            with h5py.File(self._file, 'r') as f:
                source_words = [w.encode() for w in f['words_flatten'][0].split('\n')]
                source_word_to_id = dict(zip(source_words, list(range(len(source_words)))))

                # TODO(tomwesolowski):
                # I need to copy this to memory due to terribly slow slicing.
                # Try to copy embeddings to memory in batches.
                source_matrix = f['embedding'][:]

                source_vocab_ids = []
                for i in trange(self._vocab.size()):
                    word = self._vocab.id_to_word(i)
                    if word in source_word_to_id:
                        source_vocab_ids.append((source_word_to_id[word], i))

                source_vocab_ids.sort()
                source_ids, vocab_ids = map(list, zip(*source_vocab_ids))

                target_matrix[vocab_ids] = source_matrix[source_ids][:]

                logger.info("Found {} words from dictionary in embedding file. "
                            "Missing {} words.".format(
                    len(vocab_ids), self._vocab.size() - len(vocab_ids)))

        # Padding is always zero-vector.
        target_matrix[0, :] = 0

        self._cached = target_matrix
        return target_matrix

    @property
    def dim(self):
        return self._dim

    @property
    def trainable(self):
        return self._trainable

    @classmethod
    def from_config(cls, config, rng, vocabs):
        # TODO(tomwesolowski): Make config read-only.
        vocab = vocabs[config.get('vocab', 'default')]
        config_without_vocab = dict(config)
        del config_without_vocab['vocab']
        return cls(rng, vocab, **config_without_vocab)