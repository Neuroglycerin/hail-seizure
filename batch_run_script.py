#!/usr/bin/env python3

import optparse
import os
import subprocess
import warnings
import glob

def get_batch_parser():
    '''
    Generate optparse parser object for batch_run_script.py
    with the relevant options
    input:  void
    output: optparse parser
    '''
    parser = optparse.OptionParser()

    parser.add_option("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Print verbose output")

    parser.add_option("-s", "--settings_dir",
                      action="store",
                      dest="setting_dir",
                      help="Directory holding json settings files")

    parser.add_option("-j", "--cores",
                      action="store",
                      dest="cores",
                      help="Number of cores to use in batch processing")

    return parser

def get_settings_list(settings_directory, verbose=False):
    '''
    Get list of all settings file in settings directory
    input:  settings_directory - str
    output: settings_files - list of settings_files
    '''
    settings_dir_abs_path = os.path.abspath(settings_directory)
    settings_files = glob.glob(os.path.join(settings_dir_abs_path, '*.json'))

    print_verbose('##Settings directory parsed: {0} files##'.format(\
            len(settings_files)), flag=verbose)

    return settings_files

def print_verbose(string, flag=False):
    '''
    Print statement only if flag is true
    '''
    if type(flag) is not bool:
        raise ValueError("verbose flag is not bool")
    if flag:
        print(string)


def call_train_and_predict(settings_file, verbose=False):

    out = None
    err = None

    if not verbose:
        null = open(os.devnull, 'w')
        out = null
        err = null

    print_verbose('**Training {0}**'.format(settings_file), flag=verbose)
    train_retcode = subprocess.call(['./train.py', '-s', settings_file],
                                    stdout=out, stderr=err)

    if train_retcode != 0:
        warnings.warn("train.py -s {0} did not complete successfully".format(\
                settings_file))

    print_verbose('**Predicting {0}**'.format(settings_file), flag=verbose)
    predict_retcode = subprocess.call(['./predict.py', '-s', settings_file],
                                     stdout=out, stderr=err)

    if predict_retcode != 0:
        warnings.warn("predict.py -s {0} did not complete successfully".format(\
                settings_file))

    if not verbose:
        null.close()


#def run_in_parallel(settings, cores=1, verbose=False):
#    # pipe all output to dev null for parallel
#    for settings_file in settings:
#        thread = threading.Thread(target=call_train_and_predict, args=(settings))
#        thread.start()
#        pool.append(thread)
#
#    # won't work because of subprocess call blocking
#    # but if I use Popen how do I make sure predict doesn't run ahread of train
#
#    for thread in pool:
#        thread.join()


def run_in_serial(all_settings, verbose=False):
    '''
    Run train and predict on all settings files serially
    input: all_settings - list of settings files
    '''
    num_settings = len(all_settings)
    index = 0

    for setting in all_settings:
        index+=1
        print_verbose('@@Running setting {0} of {1}@@'.format(index, num_settings),
                     flag=verbose)

        call_train_and_predict(setting, verbose=verbose)

if __name__=='__main__':

    # note: if you have set relative paths in the jsons then this script must
    # be run in a directory that contains the appropriate output folders
    # namely model and output

    parser = get_batch_parser()
    (opts, args) = parser.parse_args()

    settings_list = get_settings_list(opts.setting_dir)
    run_in_serial(settings_list, verbose=opts.verbose)
    #run_in_parallel(settings_list, cores=opts.cores, verbose=opts.verbose)

