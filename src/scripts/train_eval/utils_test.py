import numpy as np
import unittest

from src.scripts.train_eval.utils import build_data_and_streams, load_pair_features, prepare_kb

class TestUtils(unittest.TestCase):

    def test_lemma_hits(self):
        config = {
            'dataset': 'snli',
            'sentence_max_length': 2,
            'pair_features_pkl_path': 'pair_features.pkl',
            'pair_features_txt_path': 'kim_data/pair_features.txt',
        }

        features = load_pair_features(config)
        sen1 = np.array([['potassium', 'claver']])
        sen2 = np.array([['zinc', 'discourse']])

        kb_x, kb_y, hits, misses = prepare_kb(config, features, sen1, sen2)
        kb_x = np.squeeze(kb_x)
        kb_y = np.squeeze(kb_y)
        kb_x_expected = np.array([[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                                  [[0, 0, 0, 0, 0], [0.875, 0, 0, 0, 0]]])

        kb_y_expected = np.array([[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                                  [[0, 0, 0, 0, 0], [0, 0.875, 0, 0, 0]]])

        np.testing.assert_allclose(kb_x, kb_x_expected)
        np.testing.assert_allclose(kb_y, kb_y_expected)
        self.assertEqual(hits, 4)


    def test_full_lemma_hits(self):
        config = {
            'dataset': 'snli',
            'sentence_max_length': 2,
            'pair_features_pkl_path': 'pair_features.pkl',
            'pair_features_txt_path': 'kim_data/pair_features.txt',
        }
        data_streams = build_data_and_streams()

if __name__ == '__main__':
    unittest.main()