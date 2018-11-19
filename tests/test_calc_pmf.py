"""
Tests for run_mbar.py
"""
import sys
import numpy as np
import os
import pandas as pd
import unittest
from mbar_pmf.calc_pmf import main
from libs.common import capture_stdout, capture_stderr, diff_lines, silent_remove

WHAM_PATH = 'test_data/wham_pmf.csv'
MBAR_PATH_WRITE = 'mbar_pmf_test.txt'
MBAR_PATH_READ = os.path.join('test_data', MBAR_PATH_WRITE)


class TestMBAR(unittest.TestCase):
    def testOutput(self):
        test_input = ['test_data', '-o {}'.format(MBAR_PATH_WRITE)]
        main(test_input)

        self.assertTrue(os.path.isfile(WHAM_PATH))
        self.assertTrue(os.path.isfile(MBAR_PATH_READ))
        df_wham = pd.read_table(WHAM_PATH, comment='#', usecols=[0, 1], skiprows=2, names=['dist', 'pmf'])
        df_mbar = pd.read_table(MBAR_PATH_READ, delimiter=',', comment='#', skiprows=2, names=['dist', 'pmf', 'error'])
        self.assertTrue(np.allclose(df_wham.pmf, df_mbar.pmf, rtol=0.3, atol=0.3))


class TestFailWell(unittest.TestCase):
    def testNoArgs(self):
        test_input = []
        main(test_input)

    def testHelp(self):
        test_input = ["-h"]

        main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testBadDataPath(self):
        test_input = ['test_dat']
        main(test_input)
        with capture_stderr(main, test_input) as output:
            print('output is', output)
            self.assertTrue("does not exist" in output)