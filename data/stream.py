from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from keras.utils.np_utils import to_categorical

from data.transformers import NLITransformer, NLIBatchTransformer, NLIShuffleTransformer


class NLIStream(object):
    def __init__(self, dataset, rng, batch_transformers,
                 fraction, batch_size, shuffle):
        self._dataset = dataset
        self._rng = rng
        self._fraction = fraction
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._transformers = []
        self._force_reset = False

        self._epoch_iterator = None

        if self._shuffle:
            self._transformers.append(NLIShuffleTransformer(rng=rng,
                                                            batch_size=self._batch_size))
        self._transformers.append(NLIBatchTransformer(self._batch_size))
        self._transformers.extend(batch_transformers)

        self._stream = self._build()

    @property
    def num_examples(self):
        return int(self._fraction * self._dataset.num_examples)

    def _build(self):
        scheme = self._get_fraction_scheme()
        stream = DataStream(self._dataset, iteration_scheme=scheme)

        for transformer in self._transformers:
            stream = transformer.transform(stream)

        return stream

    def _get_fraction_scheme(self):
        if self._fraction < 1.0:
            # Take random subset of dataset.
            return SequentialExampleScheme(
                examples=self._rng.choice(
                    a=self._dataset.num_examples,
                    size=self.num_examples,
                    replace=False
                )
            )
        return SequentialExampleScheme(self.num_examples)

    def __len__(self):
        """
        :return: number of batches per epochs
        """
        return (self.num_examples + self._batch_size - 1) // self._batch_size

    def _next_batch(self):
        if self._epoch_iterator is None:
            self.reset()
        try:
            batch = next(self._epoch_iterator)
        except StopIteration:
            self.reset()
            batch = next(self._epoch_iterator)
        return batch

    def __next__(self):
        batch = self._next_batch()
        y = batch['label']
        del batch['label']
        return batch, to_categorical(y, 3)

    def reset(self):
        self._epoch_iterator = self._stream.get_epoch_iterator(as_dict=True)

    # TODO(tomwesolowski): Simplify list of arguments by introducing Experiment class.
    @classmethod
    def from_config(cls, config, dataset, rng, batch_transformers,):
        return cls(dataset=dataset,
                   rng=rng,
                   batch_transformers=batch_transformers,
                   **config)