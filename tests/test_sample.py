# Sample Test passing with nose and pytest
import unittest
from mbar_pmf.calc_pmf import ReturnButt


class TestDummyButt(unittest.TestCase):
    def testReturnButt(self):
        self.assertEqual(ReturnButt('dummy'), 'butt')


def test_pass():
        assert True, "dummy sample test"
