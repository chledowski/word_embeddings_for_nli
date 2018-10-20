import pathlib
import unittest

from common.utils import prepare_environment


class NLITestCase(unittest.TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self):
        self._rng = prepare_environment({
            "seed": 123
        })