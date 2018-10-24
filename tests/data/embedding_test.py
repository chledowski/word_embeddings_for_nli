import json
import logging
import numpy as np
import numpy.testing as npt
import os
import sys
import unittest

from data.vocabulary import NLIVocabulary
from data.embedding import NLIEmbedding
from tests.common.test_case import NLITestCase

logger = logging.getLogger(__name__)


class EmbeddingsTestCase(NLITestCase):

    def setUp(self):
        super(EmbeddingsTestCase, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / "embedding"

        with open(self.TEST_FIXTURES_ROOT / "config.json") as f:
            config = json.load(f)

        vocab_config = config['vocabs']['test']
        vocab_config['file_or_data'] = os.path.join(self.TEST_FIXTURES_ROOT,
                                                    vocab_config['file_or_data'])
        self.vocab = NLIVocabulary.from_config(config=vocab_config)

        emb_config = config['embeddings']['test']
        emb_config['file'] = os.path.join(self.TEST_FIXTURES_ROOT,
                                          emb_config['file'])
        self.embedding = NLIEmbedding.from_config(config=emb_config,
                                                  rng=self._rng,
                                                  vocabs={
                                                      'test': self.vocab
                                                  })

    def test_random(self):
        target_matrix_one = self.embedding.load()
        target_matrix_two = self.embedding.load(force_reload=True)
        self.assertFalse(np.allclose(target_matrix_one[2],
                                     target_matrix_two[2]))  # word out of vocab

    def test_cache(self):
        target_matrix_one = self.embedding.load()
        target_matrix_two = self.embedding.load()
        self.assertTrue(np.allclose(target_matrix_one[2],
                                    target_matrix_two[2]))  # word out of vocab

    def test_full(self):
        target_matrix = self.embedding.load()
        self.assertEqual((4, 2), target_matrix.shape)
        npt.assert_allclose([0.0, 0.0], target_matrix[0])  # padding
        npt.assert_allclose([0.5, 0.5], target_matrix[1])
        npt.assert_allclose([0.0, 1.0], target_matrix[3])
        self.assertEqual(0, self.vocab.pad)


if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        stream=sys.stdout)

    unittest.main()
