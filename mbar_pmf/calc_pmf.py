"""
Calculates PMF for given signac statepoints using MBAR and writes to file. Modified from example included in
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymbar
from pymbar import timeseries
import sys
from libs.common import warning


DEF_OUTPUT_FILE = 'mbar_pmf.txt'
GOOD_RET = 0
INPUT_ERROR = 1
IO_ERROR = 2
INVALID_DATA = 3
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='Performs MBAR analysis on a given data set.')

    parser.add_argument("-o", "--output_file",
                        help="Location of the dictionary file to be modified. "
                             "The default is: '{}'".format(DEF_OUTPUT_FILE), default=DEF_OUTPUT_FILE)

    parser.add_argument("data_dir", help="path to the data directory")

    args = None
    try:
        args = parser.parse_args(argv)
    except (IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    if not os.path.exists(args.data_dir):
        warning('The data directory "{}/{}" does not exist!'.format(__file__, args.data_dir))
        return args, INVALID_DATA

    return args, GOOD_RET


def mbar(workspace_path, output_name=DEF_OUTPUT_FILE):
    print('initializing MBAR')
    # kB = 1.0  # reduced units
    # temperature = 1  # assume a single temperature in reduced units of kT
    n_max = 50000  # maximum number of snapshots per simulation
    total_windows = 140
    n_windows = 140
    n_bins = 300
    k = 100
    r0_min = 127
    r0_max = 155
    n_rows_to_skip = 0
    # Allocate storage for simulation data_dump
    N_k = np.zeros([n_windows], dtype=int)  # number of data points from umbrella window k
    K_k = np.ones([n_windows]) * k  # spring constant for all umbrella simulations
    r0_k = np.linspace(r0_min, r0_max, total_windows)[:n_windows]
    r0_min = min(r0_k)
    r0_max = max(r0_k)
    r_kn = np.zeros([n_windows, n_max])  # inter-particle distance for snapshot n in window k
    u_kn = np.zeros([n_windows, n_max])  # reduced pot. energy w/o umbrella restraints
    g_k = np.zeros([n_windows])

    print('reading in files ... ')
    plt.figure()
    for window, r0 in enumerate(r0_k):
        # Read in simulation data
        # filename = os.path.join(workspace_path, 'dist{:.2f}.log'.format(r0))
        filename = os.path.join(workspace_path, 'dist{}.log'.format(round(r0, 9)))
        dists = pd.read_csv(filename, delimiter='\t', usecols=[1], skiprows=n_rows_to_skip).values

        N_k[window] = len(dists)
        r_kn[window, 0:N_k[window]] = dists[:, 0]

        # Determine correlation time
        g_k[window] = timeseries.statisticalInefficiency(r_kn[window, 0:N_k[window]])
        print("Correlation time for set %5.2d is %10.3f" % (r0, g_k[window]))

        # Subsample data
        indices = timeseries.subsampleCorrelatedData(r_kn[window, 0:N_k[window]], g=g_k[window])
        print('number of data points for this window is ', len(indices))
        N_k[window] = len(indices)
        u_kn[window, 0:N_k[window]] = u_kn[window, indices]
        r_kn[window, 0:N_k[window]] = r_kn[window, indices]

    # Shorten the array size
    n_max = np.max(N_k)
    u_kln = np.zeros([n_windows, n_windows, n_max], np.float64)
    u_kn -= u_kn.min()  # Set zero of u_kn -- this is arbitrary.

    print('Binning data ... ')
    # construct bins
    delta = (r0_max-r0_min)/float(n_bins)

    # compute bin centers
    bin_center_i = np.zeros([n_bins], np.float64)

    for i in range(n_bins):
        bin_center_i[i] = r0_min + delta/2 + delta * i

    # Bin data
    bin_kn = np.zeros([n_windows, n_max], np.int32)
    for window in range(n_windows):
        for n in range(N_k[window]):
            # compute bin assignment
            bin_kn[window, n] = int((r_kn[window, n] - r0_min)/delta)

    # Evaluate reduced energies in all umbrellas
    print("evaluating reduced potential energies ... ")
    for window in range(n_windows):
        for n in range(N_k[window]):
            dr = r_kn[window, n] - r0_k
            # compute energy of snapshot n from simulation k in umbrella potential l
            u_kln[window, :, n] = u_kn[window, n] + (K_k/2.0) * dr**2

    # Initialize MBAR
    # print(u_kln)
    print("running MBAR ...")
    mbar = pymbar.MBAR(u_kln, N_k, verbose=True)
    f_i, df_i = mbar.computePMF(u_kn, bin_kn, n_bins)
    output_path = os.path.join(workspace_path, output_name)
    np.savetxt(output_path, np.c_[bin_center_i, f_i, df_i], fmt='%.8f', delimiter=',', header="r_0, f_i, df_i")
    print('written to ', output_path)


def main(argv=None):
    args, ret = parse_cmdline(argv)
    if ret != 0 or args is None:
        return ret

    mbar(args.data_dir, args.output_file)

    return 0  # success

if __name__ == '__main__':
    status = main()
    sys.exit(status)
