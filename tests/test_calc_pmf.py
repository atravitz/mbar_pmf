"""
Tests for run_mbar.py
"""
import sys
sys.path.append('../')
import numpy as np
import os
import pandas as pd
import unittest
from mbar_pmf.calc_pmf import main
from mbar_pmf.calc_pmf import mbar

WHAM_PATH = 'wham_pmf.csv'
MBAR_PATH = 'mbar_pmf_test.txt'


class TestMBAR(unittest.TestCase):
    def testOutput(self):
        mbar(workspace_path='test_data',
             output_name = MBAR_PATH)

        if os.path.isfile(WHAM_PATH) and os.path.isfile(MBAR_PATH):
            diam = 120
            df_wham = pd.read_table(WHAM_PATH, comment='#', usecols=[0, 1], skiprows=2, names=['dist', 'pmf'])
            df_mbar = pd.read_table(MBAR_PATH, delimiter=',', comment='#', skiprows=2, names=['dist', 'pmf', 'error'])
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