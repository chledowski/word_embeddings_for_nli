import copy
import pprint

from common.paths import *
from data.dataset import NLIData
from data.embedding import NLIEmbedding
from data.stream import NLIStream
from data.transformers import NLITransformer
from data.vocabulary import NLIVocabulary

from models.esim import esim


class Experiment:

    def __init__(self, config, model, dataset, vocabs, embeddings, batch_transformers, streams):
        self.config = config
        self.dataset = dataset
        self.vocabs = vocabs
        self.embeddings = embeddings
        self.batch_transformers = batch_transformers
        self.streams = streams
        self.model = model

    def print_configs(self):
        configs_to_print = [
            'dataset',
            'embeddings',
            'streams',
            'trainer',
            'model'
        ]
        for name in configs_to_print:
            print("----- %s -----" % name)
            pprint.pprint(self.config[name])

    @classmethod
    def from_config(cls, config, rng):
        # 0. Load dataset
        dataset = NLIData.from_config(config['dataset'])

        # 1. Load vocabularies
        vocabs = {}
        for name, vocab_config in config['vocabs'].items():
            vocab_config['file_or_data'] = os.path.join(dataset.path, vocab_config['file_or_data'])
            vocabs[name] = NLIVocabulary.from_config(config=vocab_config)

        # 2. Load embeddings
        embeddings = {}
        for name, emb_config in config['embeddings'].items():
            emb_config['file'] = os.path.join(EMBEDDINGS_DIR, emb_config['file'])
            embeddings[name] = NLIEmbedding.from_config(config=emb_config,
                                                        rng=rng,
                                                        vocabs=vocabs)

        # 3. Batch transformers
        batch_transformers = []
        for bt_config in config['batch_transformers']:
            bt_config = copy.deepcopy(bt_config)
            if 'vocab' in bt_config:
                bt_config['vocab'] = vocabs.get(bt_config['vocab'])
            transformer = NLITransformer.by_name(bt_config['name']).from_config(bt_config)
            batch_transformers.append(transformer)

        class StreamRegistry(dict):
            def __init__(self):
                super(StreamRegistry, self).__init__()
                self.__dict__ = self

        # 4. Load streams
        streams = StreamRegistry()
        for name in dataset.parts:
            streams[name] = NLIStream.from_config(
                config=config['streams'][name],
                dataset=dataset.part(name),
                rng=rng,
                batch_transformers=batch_transformers)

        # TODO(tomwesolowski): Read from config.
        model = esim(config=config['model'],
                     embeddings=embeddings)

        return cls(
            config=config,
            model=model,
            dataset=dataset,
            vocabs=vocabs,
            embeddings=embeddings,
            batch_transformers=batch_transformers,
            streams=streams
        )
