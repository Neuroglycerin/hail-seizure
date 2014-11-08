#!/usr/bin/env python3

import subprocess
import warnings
import os
import sys
import python.utils as utils

def call_train_and_predict(settings_file, verbose=False):


    settings = utils.get_settings(settings_file)

    batch_out_dir = 'batch_out'
    out_file = open(os.path.join(batch_out_dir,
                                 "{0}_AUC".format(settings['RUN_NAME'])),'w')

    null = open(os.devnull, 'w')
    err = null
    out = null

    train_retcode = subprocess.call(['./train.py', '-s', settings_file],
                                    stdout=out_file, stderr=err)

    # Raise a warning if it was non-zero and return
    if train_retcode != 0:
        warnings.warn("train.py -s {0} did not complete successfully".format(\
                settings_file))
        return None

    # Start ./predict proc
    predict_retcode = subprocess.call(['./predict.py', '-s', settings_file],
                                     stdout=out, stderr=err)

   # Raise warning if predict failed and return
    if predict_retcode != 0:
        warnings.warn("predict.py -s {0} did not complete successfully".format(\
                settings_file))
        return None

    return None
    null.close()
    out_file.close()

if __name__=='__main__':

    call_train_and_predict(sys.argv[1])
