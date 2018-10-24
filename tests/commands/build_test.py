import unittest

from common.experiment import Experiment
from common.utils import load_config, prepare_environment
from tests.common.test_case import NLITestCase


class BuildTest(NLITestCase):

    def setUp(self):
        super(BuildTest, self).setUp()

        self.TEST_FIXTURES_ROOT = self.FIXTURES_ROOT / "build"
        self.files = {
            "config": "esim-snli.json",
        }

        self.config = load_config(self.TEST_FIXTURES_ROOT / self.files['config'])

    def test_build(self):
        rng = prepare_environment(self.config)
        Experiment.from_config(self.config, rng=rng)


if __name__ == '__main__':
    unittest.main()