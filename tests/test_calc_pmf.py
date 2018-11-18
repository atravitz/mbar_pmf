"""
Tests for run_mbar.py
"""
import sys
sys.path.append('../')
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import unittest
from mbar_pmf.calc_pmf import mbar
# logging.basigConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

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
