from fuel.datasets import Dataset
from fuel.transformers import Transformer, AgnosticSourcewiseTransformer
from picklable_itertools import iter_, chain


class FixedMapping(Transformer):
    """Applies a mapping to the data of the wrapped data stream.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    mapping : callable
        The mapping to be applied.
    add_sources : tuple of str, optional
        When given, the data produced by the mapping is added to original
        data under source names `add_sources`.

    """
    def __init__(self, data_stream, mapping, add_sources=None, **kwargs):
        super(FixedMapping, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.mapping = mapping
        self.add_sources = add_sources

    @property
    def sources(self):
        return self.data_stream.sources + (self.add_sources
                                           if self.add_sources else ())

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        image = self.mapping(self.data_stream.sources, data)
        if not self.add_sources:
            return image
        # This is the fixed line. We need to transform data to list(data) to concatenate the two!
        return tuple(list(data) + image)


class SourcewiseMapping(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, mapping, *args, **kwargs):
        kwargs.setdefault('which_sources', data_stream.sources)
        super(SourcewiseMapping, self).__init__(
            data_stream, data_stream.produces_examples, *args, **kwargs)
        self._mapping = mapping

    def transform_any_source(self, source_data, _):
        return self._mapping(source_data)


class TextFile(Dataset):
    def __init__(self, files, sources):
        self.files = files
        self.provides_sources = sources
        super(TextFile, self).__init__(sources=sources)

    def open(self):
        handlers = [open(f) for f in self.files]
        return chain(*[iter_(h) for h in handlers]), handlers

    def close(self, state):
        data, handlers = state
        for h in handlers:
            h.close()

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        data, handlers = state
        line = next(data)
        return tuple(line.split())