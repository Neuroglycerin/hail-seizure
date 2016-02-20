#!/usr/bin/env python3

import optparse
import time
import sys
import os
import subprocess
import warnings
import glob
from multiprocessing import Process

import utils
import train
from train_and_predict import call_train_and_predict


def get_batch_parser():
    '''
    Generate optparse parser object for batch_run_script.py
    with the relevant options
    input:  void
    output: optparse parser
    '''
    parser = optparse.OptionParser()

    parser.add_option(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print verbose output",
    )
    parser.add_option(
        "-s", "--settings_dir",
        action="store",
        dest="setting_dir",
        help="Directory holding json settings files",
    )
    parser.add_option(
        "-j", "--cores",
        action="store",
        dest="cores",
        default=3,
        help="Number of cores to use in batch processing",
    )
    parser.add_option(
        "--nopredict",
        action="store_false",
        dest="dopredict",
        default=True,
        help="Use this to run the train step without generating predictions",
    )

    return parser


def get_settings_list(settings_directory, verbose=False):
    '''
    Get list of all settings file in settings directory
    input:  settings_directory - str
    output: settings_files - list of settings_files
    '''
    settings_dir_abs_path = os.path.abspath(settings_directory)
    settings_files = glob.glob(os.path.join(settings_dir_abs_path, '*.json'))

    print_verbose('##Settings directory parsed: {0} files##'.format(
        len(settings_files)), flag=verbose)

    return settings_files


def print_verbose(string, flag=False):
    '''
    Print statement only if flag is true
    '''
    if not isinstance(flag, bool):
        raise ValueError("verbose flag is not bool")
    if flag:
        print(string)


def batch_run(settings_file, dopredict=True, verbose=False):

    print_verbose("==Running {0}==".format(settings_file), flag=verbose)

    try:
        if dopredict:
            call_train_and_predict(settings_file)
        else:
            train.main(settings_file)
        print_verbose(
            "**Finished {0}**".format(settings_file),
            flag=verbose)

    except:
        warnings.warn("File {0} did not complete successfully".format(
            settings_file))

    sys.stdout.flush()


if __name__ == '__main__':

    parser = get_batch_parser()
    (opts, args) = parser.parse_args()

    settings_list = get_settings_list(opts.setting_dir)

    fn = lambda x: batch_run(x, dopredict=opts.dopredict, verbose=opts.verbose)
    with Pool(processes=int(opts.cores)) as pool:
        pool.apply(fn, args=settings_list)
