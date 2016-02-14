#!/usr/bin/env python3

import subprocess
import warnings
import os
import sys

import utils


def call_train_and_predict(settings_file, verbose=False):

    settings = utils.get_settings(settings_file)

    null = open(os.devnull, 'w')

    train_retcode = subprocess.call(['./train.py', '-s', settings_file],
                                    stdout=null, stderr=null)

    # Raise a warning if it was non-zero and return
    if train_retcode != 0:
        warnings.warn("train.py -s {0} did not complete successfully".format(
            settings_file))
        return None

    # Start ./predict proc
    predict_retcode = subprocess.call(['./predict.py', '-s', settings_file],
                                      stdout=null, stderr=null)

    # Raise warning if predict failed and return
    if predict_retcode != 0:
        warnings.warn("predict.py -s {0} did not complete successfully".format(
            settings_file))
        return None

    return None
    null.close()
    out_file.close()

if __name__ == '__main__':

    call_train_and_predict(sys.argv[1])
