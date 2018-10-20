import math
import numpy

from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, Unpack

from common.registrable import Registrable
from data.utils import FixedMapping, SourcewiseMapping


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
