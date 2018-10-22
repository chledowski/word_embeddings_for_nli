import logging
import math
import numpy
import pickle

from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, Unpack

from common.paths import *
from common.registrable import Registrable
from data.utils import FixedMapping, SourcewiseMapping

logger = logging.getLogger(__name__)


class NLITransformer(Registrable):
    def __init__(self, **kwargs):
        pass

    def transform(self, stream):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config, **kwargs)


@NLITransformer.register('batcher')
class NLIBatchTransformer(NLITransformer):
    def __init__(self, batch_size, **kwargs):
        super(NLIBatchTransformer, self).__init__(**kwargs)
        self._batch_size = batch_size

    def transform(self, stream):
        return Batch(stream, iteration_scheme=ConstantScheme(self._batch_size))


@NLITransformer.register('shuffler')
class NLIShuffleTransformer(NLITransformer):
    def __init__(self, rng, batch_size, sort_by='sentence2', **kwargs):
        super(NLIShuffleTransformer, self).__init__(**kwargs)
        # TODO(tomwesolowski): Add to config.
        self._batch_size = batch_size
        self._buffer_size = 20 * batch_size
        self._sort_by = sort_by
        self._rng = rng

    def transform(self, stream):
        stream = Batch(stream, iteration_scheme=ConstantScheme(self._buffer_size))
        stream = FixedMapping(stream, self._shuffle)
        return Unpack(stream)

    def _shuffle(self, sources, data):
        # TODO(tomwesolowski): Clean this code.

        sort_by_index = sources.index(self._sort_by)
        target = data[sort_by_index]
        target_lens = numpy.array([len(t) for t in target])
        sorted_indices = target_lens.argsort(kind='mergesort')

        batch_index = numpy.array(list(
            range(int(math.ceil(len(target) * 1. / self._batch_size)))
        ))
        self._rng.shuffle(batch_index)

        final_indices = []
        for i in batch_index:
            if (i + 1) * self._batch_size > len(target):
                final_indices.extend(sorted_indices[i * self._batch_size:])
            else:
                final_indices.extend(sorted_indices[i * self._batch_size:(i + 1) * self._batch_size])

        data = [d[final_indices] for d in data]
        return data


@NLITransformer.register('indexer')
class NLIIndexTransformer(NLITransformer):
    def __init__(self, vocab, **kwargs):
        super(NLIIndexTransformer, self).__init__(**kwargs)
        self._vocab = vocab

    def transform(self, stream):
        return SourcewiseMapping(stream, self._digitize, which_sources=('sentence1', 'sentence2'))

    def _digitize(self, source_data):
        return numpy.array([self._vocab.encode(words) for words in source_data])


@NLITransformer.register('padder')
class NLIPadTransformer(NLITransformer):
    def __init__(self, **kwargs):
        super(NLIPadTransformer, self).__init__(**kwargs)

    def transform(self, stream):
        return Padding(stream, mask_sources=('sentence1', 'sentence2'))


@NLITransformer.register('wordnet')
class WordNetTransformer(NLITransformer):
    def __init__(self, file, **kwargs):
        super(WordNetTransformer, self).__init__(**kwargs)
        self.pair_features = self._load_pair_features(file)

    def transform(self, stream):
        return FixedMapping(stream, self._add_kb,
                            add_sources=('KBph', 'KBhp'))

    def _add_kb(self, sources, data):
        x1_lemma = data[sources.index('sentence1_lemmatized')]
        x2_lemma = data[sources.index('sentence2_lemmatized')]
        return self._prepare_kb(x1_lemma, x2_lemma)

    # TODO(tomwesolowski): Clean this code.
    def _prepare_kb(self, x1_lemma, x2_lemma):
        batch_size = x1_lemma.shape[0]
        x1_length = numpy.max([s.shape[0] for s in x1_lemma])
        x2_length = numpy.max([s.shape[0] for s in x2_lemma])
        kb_x = numpy.zeros((batch_size, x1_length, x2_length, 5)).astype('float32')
        kb_y = numpy.zeros((batch_size, x2_length, x1_length, 5)).astype('float32')

        total_misses = 0
        total_hits = 0

        def fill_kb(batch_id, words1, words2, kb):
            hits, misses = 0, 0
            for i1 in range(len(words1)):
                w1 = words1[i1].decode()
                for i2 in range(len(words2)):
                    w2 = words2[i2].decode()
                    if w1 in self.pair_features and w2 in self.pair_features[w1]:
                        kb[batch_id][i1][i2] = self.pair_features[w1][w2]
                        hits += 1
                    else:
                        misses += 1
            return hits, misses

        for batch_id in range(batch_size):
            h, m = fill_kb(batch_id, x1_lemma[batch_id], x2_lemma[batch_id], kb_x)
            total_hits += h
            total_misses += m
            h, m = fill_kb(batch_id, x2_lemma[batch_id], x1_lemma[batch_id], kb_y)
            total_hits += h
            total_misses += m

        return [kb_x, kb_y]

    def _load_pair_features(self, path):
        full_path = os.path.join(DATA_DIR, path)
        with open(full_path, 'rb') as f:
            features = pickle.load(f)
        logger.info("Loaded word-net features from: %s" % full_path)
        return features
