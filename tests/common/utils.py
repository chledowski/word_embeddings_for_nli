import pathlib
import unittest

from common.utils import prepare_environment


class RandomState(object):

    def shuffle(self, x):
        return x