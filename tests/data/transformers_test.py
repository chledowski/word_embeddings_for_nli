import json
import unittest

from fuel.streams import DataStream

from data.transformers import NLIShuffleTransformer
from data.utils import TextFile
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
        with open(self.TEST_FIXTURES_ROOT / self.files['expected']) as f:
            for line in f:
                self.expected.append(tuple(line.split()))

    def tearDown(self):
        self.stream.close()

    def test_shuffler(self):
        transformed_stream = self.transformer.transform(self.stream)
        data = list(transformed_stream.get_epoch_iterator())
        self.assertEqual(data, self.expected)


if __name__ == '__main__':
    unittest.main()