import h5py
import logging

from fuel.datasets import H5PYDataset

from common.paths import *
from common.registrable import Registrable

logger = logging.getLogger(__name__)


class NLIData(Registrable):
    def __init__(self, name, part_paths, use_lemmatized, sources, lemmatized_sources):
        self._name = name
        self._dataset_cache = {}
        self._path = os.path.join(DATA_DIR, self.name)
        self._part_paths = dict(zip(
            part_paths.keys(), [os.path.join(self._path, p) for p in part_paths.values()]
        ))
        self._sources = lemmatized_sources if use_lemmatized else sources

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def parts(self):
        return self._part_paths.keys()

    @property
    def evaluation_parts(self):
        raise NotImplementedError()

    def part(self, part):
        if part not in self._dataset_cache:
            self._dataset_cache[part] = H5PYDataset(
                h5py.File(self._part_paths[part], "r"),
                which_sets=('all',),
                sources=self._sources,
                load_in_memory=True
            )
        return self._dataset_cache[part]

    @classmethod
    def from_config(cls, config):
        # pylint: disable=protected-access
        return cls.by_name(config['name'])._from_config(config)

    @classmethod
    def _from_config(cls, config):
        return cls(**config)


@NLIData.register('snli')
class SNLIData(NLIData):
    def __init__(self, **kwargs):
        sources = ('sentence1', 'sentence2', 'label',)
        lemmatized_sources = (
            'sentence1', 'sentence1_lemmatized',
            'sentence2', 'sentence2_lemmatized',
            'label'
        )
        part_paths = {
            'train': 'train.h5',
            'dev': 'dev.h5',
            'test': 'test.h5',
            'breaking': 'breaking.h5'
        }
        super(SNLIData, self).__init__(
            part_paths=part_paths,
            sources=sources,
            lemmatized_sources=lemmatized_sources,
            **kwargs
        )

    @property
    def evaluation_parts(self):
        return ['dev', 'test', 'breaking']


@NLIData.register('mnli')
class MNLIData(NLIData):
    def __init__(self, **kwargs):
        sources = ('sentence1', 'sentence2', 'label',)
        lemmatized_sources = (
            'sentence1', 'sentence1_lemmatized',
            'sentence2', 'sentence2_lemmatized',
            'label'
        )
        part_paths = {
            'train': 'train.h5',
            'dev': 'dev.h5',
            'test': 'dev_mismatched.h5'
        }
        super(MNLIData, self).__init__(
            part_paths=part_paths,
            sources=sources,
            lemmatized_sources=lemmatized_sources,
            **kwargs
        )

    @property
    def evaluation_parts(self):
        return ['dev', 'test']