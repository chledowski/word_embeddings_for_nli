import json
import numpy
import unittest

from fuel.streams import DataStream

from data.transformers import NLITransformer, NLIBatchTransformer, NLIShuffleTransformer
from data.utils import TextFile
from data.vocabulary import NLIVocabulary

from tests.common.test_case import NLITestCase
from tests.common.utils import RandomState


class ShufflerTest(NLITestCase):

    def setUp(self):
        super(ShufflerTest, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / 'shuffler'
        self.files = {
            "config": "config.json",
            "datasets": ["dataset.txt"],
            "expected": "dataset_expected.txt"
        }
        with open(self.TEST_FIXTURES_ROOT / self.files['config'], 'r') as f:
            self.config = json.load(f)

        self.dataset = TextFile([self.TEST_FIXTURES_ROOT / f for f in self.files['datasets']],
                                sources=('one', 'two', 'three'))
        self.stream = DataStream(self.dataset)
        self.transformer = NLIShuffleTransformer(rng=RandomState(), **self.config)

        self.expected = []
        with open(self.TEST_FIXTURES_ROOT / self.files['expected'], 'rb') as f:
            for line in f:
                self.expected.append(
                    tuple(sentence.split() for sentence in line.split(b',')))

    def tearDown(self):
        self.stream.close()

    def test_shuffler(self):
        transformed_stream = self.transformer.transform(self.stream)
        data = list(transformed_stream.get_epoch_iterator())
        self.assertEqual(data, self.expected)


class IndexerTest(NLITestCase):

    def setUp(self):
        super(IndexerTest, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / 'indexer'
        self.files = {
            "config": "config.json",
            "datasets": ["dataset.txt"],
            "expected": "dataset_expected.txt"
        }

        # Load transformer config
        with open(self.TEST_FIXTURES_ROOT / self.files['config'], 'r') as f:
            self.config = json.load(f)
        transformer_config = self.config['indexer']

        # Load vocab from config
        vocab_config = self.config['vocabs']['test']
        vocab_config['file_or_data'] = str(
                self.TEST_FIXTURES_ROOT / vocab_config['file_or_data'])
        transformer_config['vocab'] = NLIVocabulary.from_config(config=vocab_config)

        # Load transformer
        self.transformer = NLITransformer.from_config(transformer_config)

        # Load test data
        self.stream = DataStream(
            TextFile([self.TEST_FIXTURES_ROOT / f for f in self.files['datasets']],
                     sources=('sentence1', 'sentence2', 'sentence3'))
        )
        # Since we test batch transformer...
        self.stream = NLIBatchTransformer(1).transform(self.stream)

        # ... and expected output.
        self.expected = []
        with open(self.TEST_FIXTURES_ROOT / self.files['expected'], 'rb') as f:
            for line in f:
                self.expected.append(
                    tuple([sentence.split()] for sentence in line.split(b',')))

    def tearDown(self):
        self.stream.close()

    def _convert_to_bytes(self, data):
        if type(data) in [bytes, numpy.bytes_]:
            return data
        if type(data) is str:
            return data.encode()
        try:
            iter(data)
        except TypeError:
            return str(data).encode()
        else:
            datatype = type(data)
            if datatype == numpy.ndarray:
                datatype = numpy.array
            x = [self._convert_to_bytes(d) for d in data]
            return datatype(x)

    def test_all(self):
        transformed_stream = self.transformer.transform(self.stream)
        data = list(transformed_stream.get_epoch_iterator())
        numpy.testing.assert_equal(self._convert_to_bytes(data), self.expected)


if __name__ == '__main__':
    unittest.main()