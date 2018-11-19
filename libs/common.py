#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions and constants to be used my multiple scripts.
"""

from __future__ import print_function

import collections
import csv
import difflib
import errno
import fnmatch
import numpy as np
import os
import six
import sys
from contextlib import contextmanager
import argparse
import math


__author__ = 'hbmayes'


# Constants #

# Tolerance initially based on double standard machine precision of 5 × 10−16 for float64 (decimal64)
# found to be too stringent
TOL = 0.00000000001
# similarly, use this to round away the insignificant digits!
SIG_DECIMALS = 12

# Constants #

# Error Codes
# The good status code
GOOD_RET = 0
INPUT_ERROR = 1
IO_ERROR = 2
INVALID_DATA = 3

TPL_IO_ERR_MSG = "Couldn't read template at: '{}'"
MISSING_SEC_HEADER_ERR_MSG = "Configuration files must start with a section header such as '[main]'. Check file: {}"

# Boltzmann's Constant in kcal/mol Kelvin
BOLTZ_CONST = 0.0019872041
PLANCK_CONST = 9.53707E-14
# Universal gas constant in kcal/mol K
R = 0.001985877534

XYZ_ORIGIN = np.zeros(3)


# Exceptions #

class MdError(Exception):
    pass


class InvalidDataError(MdError):
    pass


class ArgumentParserError(Exception):
    pass


class TemplateNotReadableError(Exception):
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


# noinspection SpellCheckingInspection
def warning(*objs):
    """Writes a message to stderr."""
    print("WARNING: ", *objs, file=sys.stderr)


# Test utilities

# From http://schinckel.net/2013/04/15/capture-and-test-sys.stdout-sys.stderr-in-unittest.testcase/
@contextmanager
def capture_stdout(command, *args, **kwargs):
    # pycharm doesn't know six very well, so ignore the false warning
    # noinspection PyCallingNonCallable
    out, sys.stdout = sys.stdout, six.StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


@contextmanager
def capture_stderr(command, *args, **kwargs):
    # pycharm doesn't know six very well, so ignore the false warning
    # noinspection PyCallingNonCallable
    err, sys.stderr = sys.stderr, six.StringIO()
    command(*args, **kwargs)
    sys.stderr.seek(0)
    yield sys.stderr.read()
    sys.stderr = err


# noinspection PyTypeChecker
def diff_lines(floc1, floc2, delimiter=","):
    """
    Determine all lines in a file are equal.
    This function became complicated because of edge cases:
        Do not want to flag files as different if the only difference is due to machine precision diffs of floats
    Thus, if the files are not immediately found to be the same:
        If not, test if the line is a csv that has floats and the difference is due to machine precision.
        Be careful if one value is a np.nan, but not the other (the diff evaluates to zero)
        If not, return all lines with differences.
    @param floc1: file location 1
    @param floc2: file location 1
    @param delimiter: defaults to CSV
    @return: a list of the lines with differences
    """
    diff_lines_list = []
    # Save diffs to strings to be converted to use csv parser
    output_plus = ""
    output_neg = ""
    with open(floc1, 'r') as file1:
        with open(floc2, 'r') as file2:
            diff = list(difflib.ndiff(file1.read().splitlines(), file2.read().splitlines()))

    for line in diff:
        if line.startswith('-') or line.startswith('+'):
            diff_lines_list.append(line)
            if line.startswith('-'):
                output_neg += line[2:]+'\n'
            elif line.startswith('+'):
                output_plus += line[2:]+'\n'

    if len(diff_lines_list) == 0:
        return diff_lines_list

    warning("Checking for differences between files {} {}".format(floc1, floc2))
    try:
        # take care of parentheses
        for char in ('(', ')', '[', ']'):
            output_plus = output_plus.replace(char, delimiter)
            output_neg = output_neg.replace(char, delimiter)
        # pycharm doesn't know six very well
        # noinspection PyCallingNonCallable
        diff_plus_lines = list(csv.reader(six.StringIO(output_plus), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
        # noinspection PyCallingNonCallable
        diff_neg_lines = list(csv.reader(six.StringIO(output_neg), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
    except ValueError:
        diff_plus_lines = output_plus.split('\n')
        diff_neg_lines = output_neg.split('\n')
        for diff_list in [diff_plus_lines, diff_neg_lines]:
            for line_id in range(len(diff_list)):
                diff_list[line_id] = [x.strip() for x in diff_list[line_id].split(delimiter)]

    if len(diff_plus_lines) == len(diff_neg_lines):
        # if the same number of lines, there is a chance that the difference is only due to difference in
        # floating point precision. Check each value of the line, split on whitespace or comma
        diff_lines_list = []
        for line_plus, line_neg in zip(diff_plus_lines, diff_neg_lines):
            if len(line_plus) == len(line_neg):
                # print("Checking for differences between: ", line_neg, line_plus)
                for item_plus, item_neg in zip(line_plus, line_neg):
                    try:
                        item_plus = float(item_plus)
                        item_neg = float(item_neg)
                        # if difference greater than the tolerance, the difference is not just precision
                        # Note: if only one value is nan, the float diff is zero!
                        #  Thus, check for diffs only if neither are nan; show different if only one is nan
                        diff_vals = False
                        if np.isnan(item_neg) != np.isnan(item_plus):
                            diff_vals = True
                            warning("Comparing '{}' to '{}'.".format(item_plus, item_neg))
                        elif not (np.isnan(item_neg) and np.isnan(item_plus)):
                            # noinspection PyTypeChecker
                            if not np.isclose(item_neg, item_plus, TOL):
                                diff_vals = True
                                warning("Values {} and {} differ.".format(item_plus, item_neg))
                        if diff_vals:
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            break
                    except ValueError:
                        # not floats, so the difference is not just precision
                        if item_plus != item_neg:
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            break
            # Not the same number of items in the lines
            else:
                diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
    return diff_lines_list


def silent_remove(filename, disable=False):
    """
    Removes the target file name, catching and ignoring errors that indicate that the
    file does not exist.

    :param filename: The file to remove.
    :param disable: boolean to flag if want to disable removal
    """
    if not disable:
        try:
            os.remove(filename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


# End of test utilities; begin functions for programs

# I/O #

def read_tpl(tpl_loc):
    """Attempts to read the given template location and throws A
    TemplateNotReadableError if it can't read the given location.

    :param tpl_loc: The template location to read.
    :raise TemplateNotReadableError: If there is an IOError reading the location.
    """
    try:
        return file_to_str(tpl_loc)
    except IOError:
        raise TemplateNotReadableError(TPL_IO_ERR_MSG.format(tpl_loc))


def file_to_str(fname):
    """
    Reads and returns the contents of the given file.

    :param fname: The location of the file to read.
    @return: The contents of the given file.
    :raises: IOError if the file can't be opened for reading.
    """
    with open(fname) as f:
        return f.read()


def str_to_file(str_val, fname, mode='w', print_info=False):
    """
    Writes the string to the given file.
    :param str_val: The string to write.
    :param fname: The location of the file to write
    :param mode: default mode is to overwrite file
    :param print_info: boolean to specify whether to print action to stdout
    """
    with open(fname, mode) as f:
        f.write(str_val)
    if print_info:
        print("Wrote file: {}".format(fname))


def list_to_file(list_to_print, fname, list_format=None, delimiter=' ', mode='w', print_message=True):
    """
    Writes the list of sequences to the given file in the specified format for a PDB.

    :param list_to_print: A list of lines to print. The list may be a list of lists, list of strings, or a mixture.
    :param fname: The location of the file to write.
    :param list_format: Specified formatting for the line if the line is  list.
    :param delimiter: If no format is given and the list contains lists, the delimiter will join items in the list.
    :param print_message: boolean to determine whether to write to output if the file is printed or appended
    :param mode: write by default; can be changed to allow appending to file.
    """
    with open(fname, mode) as w_file:
        for line in list_to_print:
            if isinstance(line, six.string_types):
                w_file.write(line + '\n')
            elif isinstance(line, collections.Iterable):
                if list_format is None:
                    w_file.write(delimiter.join(map(str, line)) + "\n")
                else:
                    w_file.write(list_format.format(*line) + '\n')
    if print_message:
        if mode == 'w':
            print("Wrote file: {}".format(fname))
        elif mode == 'a':
            print("  Appended: {}".format(fname))


def find_files_by_dir(tgt_dir, pat):
    """Recursively searches the target directory tree for files matching the given pattern.
    The results are returned as a dict with a list of found files keyed by the absolute
    directory name.
    :param tgt_dir: The target base directory.
    :param pat: The file pattern to search for.
    @return: A dict where absolute directory names are keys for lists of found file names
        that match the given pattern.
    """
    match_dirs = {}
    for root, dirs, files in os.walk(tgt_dir):
        matches = [match for match in files if fnmatch.fnmatch(match, pat)]
        if matches:
            match_dirs[os.path.abspath(root)] = matches
    return match_dirs


def get_fname_root(src_file):
    """

    :param src_file:
    @return: the file root name (no directory, no extension)
    """
    return os.path.splitext(os.path.basename(src_file))[0]


def read_csv_dict(d_file, ints=True, one_to_one=True, pdb_dict=False, str_float=False, strip=False):
    """
    If an dictionary file is given, read it and return the dict[old]=new.
    Checks that all keys are unique.
    If one_to_one=True, checks that there 1:1 mapping of keys and values.

    :param d_file: the file with csv of old_id,new_id
    :param ints: boolean to indicate if the values are to be read as integers
    :param one_to_one: flag to check for one-to-one mapping in the dict
    :param pdb_dict: flag to format as required for the PDB output
    :param str_float: indicates dictionary is a string followed by a float
    :param strip: whether to strip entries after they have been split
    :return: new_dict

    """
    new_dict = {}
    if pdb_dict:
        ints = False
        one_to_one = False
    elif str_float:
        ints = False
        one_to_one = False
    # If d_file is None, return the empty dictionary, as no dictionary file was specified
    if d_file is not None:
        with open(d_file) as csv_file:
            reader = csv.reader(csv_file)
            key_count = 0
            for row in reader:
                if len(row) == 0:
                    continue
                if len(row) == 2:
                    if strip:
                        row[0] = row[0].strip()
                        row[1] = row[1].strip()
                    if ints:
                        new_dict[int(row[0])] = int(row[1])
                    elif str_float:
                        new_dict[row[0]] = float(row[1])
                    else:
                        new_dict[row[0]] = row[1]
                    key_count += 1
                else:
                    raise InvalidDataError("Error reading line '{}' in file: {}\n"
                                           "  Expected exactly two comma-separated values per row."
                                           "".format(row, d_file))
        if key_count == len(new_dict):
            if one_to_one:
                for key in new_dict:
                    if not (key in new_dict.values()):
                        raise InvalidDataError('Did not find a 1:1 mapping of key,val ids in {}'.format(d_file))
        else:
            raise InvalidDataError('A non-unique key value (first column) found in file: {}\n'.format(d_file))
    return new_dict


def read_csv_header(src_file):
    """Returns a list containing the values from the first row of the given CSV
    file or None if the file is empty.

    :param src_file: The CSV file to read.
    @return: The first row or None if empty.
    """
    with open(src_file) as csv_file:
        for row in csv.reader(csv_file):
            return list(row)


def create_out_fname(src_file, prefix='', suffix='', remove_prefix=None, base_dir=None, ext=None):
    """Creates an outfile name for the given source file.

    :param remove_prefix: string to remove at the beginning of file name
    :param src_file: The file to process.
    :param prefix: The file prefix to add, if specified.
    :param suffix: The file suffix to append, if specified.
    :param base_dir: The base directory to use; defaults to `src_file`'s directory.
    :param ext: The extension to use instead of the source file's extension;
        defaults to the `scr_file`'s extension.
    @return: The output file name.
    """

    if base_dir is None:
        base_dir = os.path.dirname(src_file)

    file_name = os.path.basename(src_file)
    if remove_prefix is not None and file_name.startswith(remove_prefix):
        base_name = file_name[len(remove_prefix):]
    else:
        base_name = os.path.splitext(file_name)[0]

    if ext is None:
        ext = os.path.splitext(file_name)[1]

    return os.path.abspath(os.path.join(base_dir, prefix + base_name + suffix + ext))


def np_float_array_from_file(data_file, delimiter=" ", header=False, gather_hist=False):
    """
    Adds to the basic np.loadtxt by performing data checks.
    :param data_file: file expected to have space-separated values, with the same number of entries per row
    :param delimiter: default is a space-separated file
    :param header: default is no header; alternately, specify number of header lines
    :param gather_hist: default is false; gather data to make histogram of non-numerical data
    @return: a numpy array or InvalidDataError if unsuccessful, followed by the header_row (None if none specified)
    """
    header_row = None
    hist_data = {}
    with open(data_file) as csv_file:
        csv_list = list(csv.reader(csv_file, delimiter=delimiter))
    if header:
        header_row = csv_list[0]

    try:
        data_array = np.genfromtxt(data_file, dtype=np.float64, delimiter=delimiter, skip_header=header)
    except ValueError:
        data_array = None
        line_len = None
        if header:
            first_line = 1
        else:
            first_line = 0
        for row in csv_list[first_line:]:
            if len(row) == 0:
                continue
            s_len = len(row)
            if line_len is None:
                line_len = s_len
            elif s_len != line_len:
                raise InvalidDataError('File could not be read as an array of floats: {}\n  Expected '
                                       'values separated by "{}" with an equal number of columns per row.\n'
                                       '  However, found {} values on the first data row'
                                       '  and {} values on the later row: "{}")'
                                       ''.format(data_file, delimiter, line_len, s_len, row))
            data_vector = np.empty([line_len], dtype=np.float64)
            for col in range(line_len):
                try:
                    data_vector[col] = float(row[col])
                except ValueError:
                    data_vector[col] = np.nan
                    if gather_hist:
                        col_key = str(row[col])
                        if col in hist_data:
                            if col_key in hist_data[col]:
                                hist_data[col][col_key] += 1
                            else:
                                hist_data[col][col_key] = 1
                        else:
                            hist_data[col] = {col_key: 1}
            if data_array is None:
                data_array = np.copy(data_vector)
            else:
                data_array = np.vstack((data_array, data_vector))
    if np.isnan(data_array).any():
        if data_array.size == 1:
            raise InvalidDataError("Data in file was not read as an array of floats. Check input, "
                                   "e.g. if the delimiter is not ('{}')".format(delimiter))
        else:
            warning("Encountered entry (or entries) which could not be converted to a float. "
                    "'nan' will be returned for the stats for that column.")
    if len(data_array.shape) == 1:
        raise InvalidDataError("File contains a vector, not an array of floats: {}\n".format(data_file))
    return data_array, header_row, hist_data


def convert_dict_line(all_conv, data_conv, line):
    s_dict = {}
    for s_key, s_val in line.items():
        if data_conv and s_key in data_conv:
            try:
                s_dict[s_key] = data_conv[s_key](s_val)
            except ValueError as e:
                warning("Could not convert value '{}' from column '{}': '{}'.  Leaving as str".format(s_val, s_key, e))
                s_dict[s_key] = s_val
        elif all_conv:
            try:
                s_dict[s_key] = all_conv(s_val)
            except ValueError as e:
                warning("Could not convert value '{}' from column '{}': '{}'.  Leaving as str".format(s_val, s_key, e))
                s_dict[s_key] = s_val
        else:
            s_dict[s_key] = s_val
    return s_dict


def read_csv(src_file, data_conv=None, all_conv=None, quote_style=csv.QUOTE_MINIMAL):
    """
    Reads the given CSV (comma-separated with a first-line header row) and returns a list of
    dicts where each dict contains a row's data keyed by the header row.

    :param src_file: The CSV to read.
    :param data_conv: A map of header keys to conversion functions.  Note that values
        that throw a TypeError from an attempted conversion are left as strings in the result.
    :param all_conv: A function to apply to all values in the CSV.  A specified data_conv value
        takes precedence.
    :param quote_style: how to read the dictionary
    @return: A list of dicts containing the file's data.
    """
    result = []
    with open(src_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, quoting=quote_style)
        for line in csv_reader:
            result.append(convert_dict_line(all_conv, data_conv, line))
    return result


def read_csv_to_dict(src_file, col_name, data_conv=None, all_conv=None):
    """
    Reads the given CSV (comma-separated with a first-line header row) and returns a
    dict of dicts indexed on the given col_name. Each dict contains a row's data keyed by the header row.

    :param src_file: The CSV to read.
    :param col_name: the name of the column to index on
    :param data_conv: A map of header keys to conversion functions.  Note that values
        that throw a TypeError from an attempted conversion are left as strings in the result.
    :param all_conv: A function to apply to all values in the CSV.  A specified data_conv value
        takes precedence.
    @return: A list of dicts containing the file's data.
    """
    result = {}
    with open(src_file) as csv_file:
        try:
            csv_reader = csv.DictReader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
            create_dict(all_conv, col_name, csv_reader, data_conv, result, src_file)
        except ValueError:
            csv_reader = csv.DictReader(csv_file)
            create_dict(all_conv, col_name, csv_reader, data_conv, result, src_file)
    return result


def create_dict(all_conv, col_name, csv_reader, data_conv, result, src_file):
    for line in csv_reader:
        val = convert_dict_line(all_conv, data_conv, line)
        if col_name in val:
            try:
                col_val = int(val[col_name])
            except ValueError:
                col_val = val[col_name]
            if col_val in result:
                warning("Duplicate values found for {}. Value for key will be overwritten.".format(col_val))
            result[col_val] = convert_dict_line(all_conv, data_conv, line)
        else:
            raise InvalidDataError("Could not find value for {} in file {} on line {}."
                                   "".format(col_name, src_file, line))


def write_csv(data, out_fname, fieldnames, extrasaction="raise", mode='w', quote_style=csv.QUOTE_NONNUMERIC,
              print_message=True, round_digits=False):
    """
    Writes the given data to the given file location.

    :param round_digits: if desired, provide decimal number for rounding
    :param data: The data to write (list of dicts).
    :param out_fname: The name of the file to write to.
    :param fieldnames: The sequence of field names to use for the header.
    :param extrasaction: What to do when there are extra keys.  Acceptable
        values are "raise" or "ignore".
    :param mode: default mode is to overwrite file
    :param print_message: boolean to flag whether to note that file written or appended
    :param quote_style: dictates csv output style
    """
    with open(out_fname, mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames, extrasaction=extrasaction, quoting=quote_style)
        if mode == 'w':
            writer.writeheader()
        if round_digits:
            for row_id in range(len(data)):
                new_dict = {}
                for key, val in data[row_id].items():
                    if isinstance(val, float):
                        new_dict[key] = round(val, round_digits)
                    else:
                        new_dict[key] = val
                data[row_id] = new_dict
        writer.writerows(data)
    if print_message:
        if mode == 'a':
            print("  Appended: {}".format(out_fname))
        elif mode == 'w':
            print("Wrote file: {}".format(out_fname))


def list_to_csv(data, out_fname, delimiter=',', mode='w', quote_style=csv.QUOTE_NONNUMERIC,
                print_message=True, round_digits=False):
    """
    Writes the given data to the given file location.
    :param data: The data to write (list of lists).
    :param out_fname: The name of the file to write to.
    :param delimiter: string
    :param mode: default mode is to overwrite file
    :param quote_style: csv quoting style
    :param print_message: boolean to allow update
    :param round_digits: boolean to affect printing output; supply an integer to round to that number of decimals
    """
    with open(out_fname, mode) as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter, quoting=quote_style)
        if round_digits:
            for row_id in range(len(data)):
                new_row = []
                for val in data[row_id]:
                    if isinstance(val, float):
                        new_row.append(round(val, round_digits))
                    else:
                        new_row.append(val)
                data[row_id] = new_row
        writer.writerows(data)
    if print_message:
        print("Wrote file: {}".format(out_fname))


def fmt_row_data(raw_data, fmt_str):
    """ Formats the values in the dicts in the given list of raw data using
    the given format string.

    *This may not be needed at all*
    Now that I'm using csv.QUOTE_NONNUMERIC, generally don't want to format floats to strings

    :param raw_data: The list of dicts to format.
    :param fmt_str: The format string to use when formatting.
    @return: The formatted list of dicts.
    """
    fmt_rows = []
    for row in raw_data:
        fmt_row = {}
        for key, raw_val in row.items():
            fmt_row[key] = fmt_str.format(raw_val)
        fmt_rows.append(fmt_row)
    return fmt_rows


def conv_raw_val(param, def_val, int_list=True):
    """
    Converts the given parameter into the given type (default returns the raw value).  Returns the default value
    if the param is None.
    :param param: The value to convert.
    :param def_val: The value that determines the type to target.
    :param int_list: flag to specify if lists should converted to a list of integers
    @return: The converted parameter value.
    """
    if param is None:
        return def_val
    if isinstance(def_val, bool):
        if param in ['T', 't', 'true', 'TRUE', 'True']:
            return True
        else:
            return False
    if isinstance(def_val, int):
        return int(param)
    if isinstance(def_val, float):
        return float(param)
    if isinstance(def_val, list):
        if int_list:
            return to_int_list(param)
        else:
            return to_list(param)
    return param


def dequote(s):
    """
    from: http://stackoverflow.com/questions/3085382/python-how-can-i-strip-first-and-last-double-quotes
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if isinstance(s, str) and len(s) > 0:
        if (s[0] == s[-1]) and s.startswith(("'", '"')):
            return s[1:-1]
    return s


def quote(s):
    """
    Converts a variable into a quoted string
    """
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        return str(s)
    return '"' + str(s) + '"'


def process_cfg(raw_cfg, def_cfg_vals=None, req_keys=None, int_list=True):
    """
    Converts the given raw configuration, filling in defaults and converting the specified value (if any) to the
    default value's type.
    :param raw_cfg: The configuration map.
    :param def_cfg_vals: dictionary of default values
    :param req_keys: dictionary of required types
    :param int_list: flag to specify if lists should converted to a list of integers
    @return: The processed configuration.

    """
    proc_cfg = {}
    for key in raw_cfg:
        if not (key in def_cfg_vals or key in req_keys):
            raise InvalidDataError("Unexpected key '{}' in configuration ('ini') file.".format(key))
    key = None
    try:
        for key, def_val in def_cfg_vals.items():
            proc_cfg[key] = conv_raw_val(raw_cfg.get(key), def_val, int_list)
        for key, type_func in req_keys.items():
            proc_cfg[key] = type_func(raw_cfg[key])
    except KeyError as e:
        raise KeyError("Missing config val for key '{}'".format(key, e))
    except Exception as e:
        raise InvalidDataError('Problem with config vals on key {}: {}'.format(key, e))

    return proc_cfg


# Conversions #

def to_int_list(raw_val):
    return_vals = []
    for val in raw_val.split(','):
        return_vals.append(int(val.strip()))
    return return_vals


def to_list(raw_val):
    return_vals = []
    for val in raw_val.split(','):
        return_vals.append(val.strip())
    return return_vals


def str_to_bool(s):
    """
    Basic converter for Python boolean values written as a str.
    :param s: The value to convert.
    @return: The boolean value of the given string.
    @raises: ValueError if the string value cannot be converted.
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))


def conv_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# Calculations #


def calc_kbt(temp_k):
    """
    Returns the given temperature in Kelvin multiplied by Boltzmann's Constant.

    :param temp_k: A temperature in Kelvin.
    @return: The given temperature in Kelvin multiplied by Boltzmann's Constant.
    """
    return BOLTZ_CONST * temp_k


def calc_k(temp, delta_gibbs):
    """
    Returns the rate coefficient calculated from Transition State Theory in inverse seconds
    :param temp: the temperature in Kelvin
    :param delta_gibbs: the change in Gibbs free energy in kcal/mol
    @return: rate coefficient in inverse seconds
    """
    return BOLTZ_CONST * temp / PLANCK_CONST * math.exp(-delta_gibbs / (R * temp))


def pbc_dist(a, b, box):
    # TODO: make a test that ensures the distance calculated is <= sqrt(sqrt((a/2)^2+(b/2)^2) + (c/2)^2)) ?
    return np.linalg.norm(pbc_calc_vector(a, b, box))


def pbc_calc_vector(a, b, box):
    """
    Finds the vectors between two points
    :param a: xyz coords 1
    :param b: xyz coords 2
    :param box: vector with PBC box dimensions
    @return: returns the vector a - b
    """
    vec = np.subtract(a, b)
    return vec - np.multiply(box, np.asarray(list(map(round, vec / box))))


def first_pbc_image(xyz_coords, box):
    """
    Moves xyz coords to the first PBC image, centered at the origin
    :param xyz_coords: coordinates to center (move to first image)
    :param box: PBC box dimensions
    @return: xyz coords (np array) moved to the first image
    """
    return pbc_calc_vector(xyz_coords, XYZ_ORIGIN, box)


def pbc_vector_avg(a, b, box):
    diff = pbc_calc_vector(a, b, box)
    mid_pt = np.add(b, np.divide(diff, 2.0))
    # mid-point may not be in the first periodic image. Make it so by getting its difference from the origin
    return pbc_calc_vector(mid_pt, np.zeros(len(mid_pt)), box)


def unit_vector(vector):
    """ Returns the unit vector of the vector.
    http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    return vector / np.linalg.norm(vector)


def vec_angle(vec_1, vec_2):
    """
    Calculates the angle between the vectors (p2 - p1) and (p0 - p1)
    Note: assumes the vector calculation accounted for the PBC
    :param vec_1: xyz coordinate for the first pt
    :param vec_2: xyz for 2nd pt
    @return: the angle in between the vectors
    """
    unit_vec_1 = unit_vector(vec_1)
    unit_vec_2 = unit_vector(vec_2)

    return np.rad2deg(np.arccos(np.clip(np.dot(unit_vec_1, unit_vec_2), -1.0, 1.0)))


def vec_dihedral(vec_ba, vec_bc, vec_cd):
    """
    calculates the dihedral angle from the vectors b --> a, b --> c, c --> d
    where a, b, c, and d are the four points
    From:
    http://stackoverflow.com/questions/20305272/
      dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    Khouli formula
    1 sqrt, 1 cross product
    :param vec_ba: the vector connecting points b --> a, accounting for pbc
    :param vec_bc: b --> c
    :param vec_cd: c --> d
    @return: dihedral angle in degrees
    """
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    vec_bc = unit_vector(vec_bc)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = vec_ba - np.dot(vec_ba, vec_bc) * vec_bc
    w = vec_cd - np.dot(vec_cd, vec_bc) * vec_bc

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(vec_bc, v), w)
    return np.degrees(np.arctan2(y, x))
