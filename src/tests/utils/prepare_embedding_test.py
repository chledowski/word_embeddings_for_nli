import json
import logging
import numpy as np
import numpy.testing as npt
import unittest

from src.tests.common.test_case import NlpTestCase
from src.tests.common.data import SimpleData
from src.util.vocab import Vocabulary

from src.util.prepare_embedding import prep_embedding_matrix


class PrepareEmbeddingsTest(NlpTestCase):

    def setUp(self):
        super(PrepareEmbeddingsTest, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / "prepare_embedding"
        self.files = {
            "config": "config.json",
            "vocab": "vocab.txt",
            "source_matrix": "three_embeddings.h5"
        }

        with open(self.TEST_FIXTURES_ROOT / self.files['config']) as f:
            self.config = json.load(f)
        self.vocab = Vocabulary(str(self.TEST_FIXTURES_ROOT / self.files['vocab']))
        self.source_matrix_path = self.TEST_FIXTURES_ROOT / self.files['source_matrix']
        self.data = SimpleData(self.config, self.vocab)

    def test_random(self):
        target_matrix_one = prep_embedding_matrix(self.config,
                                                  self.data,
                                                  self.source_matrix_path)
        target_matrix_two = prep_embedding_matrix(self.config,
                                                  self.data,
                                                  self.source_matrix_path)
        self.assertFalse(np.allclose(target_matrix_one[2],
                                     target_matrix_two[2]))  # word out of vocab

    def test_full(self):
        target_matrix = prep_embedding_matrix(self.config,
                                              self.data,
                                              self.source_matrix_path)
        self.assertEqual((4, 2), target_matrix.shape)
        npt.assert_allclose([0.0, 0.0], target_matrix[0])  # padding
        npt.assert_allclose([0.5, 0.5], target_matrix[1])
        npt.assert_allclose([0.0, 1.0], target_matrix[3])
        self.assertEqual(0, self.vocab.pad)


if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    unittest.main()
