import pathlib
import unittest


class NlpTestCase(unittest.TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    SRC_ROOT = PROJECT_ROOT / "src"
    TESTS_ROOT = SRC_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self):
        pass