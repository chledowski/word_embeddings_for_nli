import unittest

from src.tests.common.test_case import NlpTestCase
from src.util.vocab import Vocabulary


class VocabTest(NlpTestCase):

    def setUp(self):
        super(VocabTest, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / "vocab"
        self.files = {
            "vocab": "vocab.txt",
            "vocab_special": "vocab_special.txt",
        }

        self.vocab = Vocabulary(
            str(self.TEST_FIXTURES_ROOT / self.files['vocab']))
        self.vocab_special = Vocabulary(
            str(self.TEST_FIXTURES_ROOT / self.files['vocab_special']))

    def test_words(self):
        words = sorted(self.vocab.words)
        self.assertEqual(words, [b'random', b'the', b'to'])

    def test_freqs(self):
        words = self.vocab.words
        freqs = self.vocab.frequencies
        expected = {
            b'to': 100,
            b'random': 10,
            b'the': 1
        }
        for word, freq in zip(words, freqs):
            self.assertEqual(freq, expected[word])

    def test_encode(self):
        sentence = b"to unk the random".split()
        expected = [0, -1, 2, 1]
        self.assertEqual(expected, self.vocab.encode(sentence).tolist())

    def test_special(self):
        tokens = [
            'pad', 'unk', 'bos', 'eos', 'bod', 'eod'
        ]
        for id, token in enumerate(tokens):
            self.assertEqual(id, self.vocab_special.__getattribute__(token))


if __name__ == '__main__':
    unittest.main()