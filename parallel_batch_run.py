#!/usr/bin/env python3
import multiprocessing
import optparse
import time
import os
import subprocess
import python.utils as utils
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
                      default=3,
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

def batch_run_in_parallel(settings_list, cores, verbose=False):

    processes = []
    num_to_run = len(settings_list)
    finish_count = 0
    while True:
        while settings_list and len(processes) < cores:
            settings_file = settings_list.pop()
            print_verbose("==Running {0}==".format(settings_file),
                                                   flag=verbose)
            processes.append(subprocess.Popen(['./train_and_predict.py',
                                                settings_file]))

        for p in processes:
            if p.poll() is not None:
                if p.returncode == 0:
                    processes.remove(p)
                    finish_count += 1
                    print_verbose("**Completed {0} of {1}**".format(finish_count,
                                                                    num_to_run),
                                  flag=verbose)
                else:
                    sys.exit(1)

        if not processes and not settings_list:
            break
        else:
            time.sleep(0.05)


if __name__=='__main__':


    parser = get_batch_parser()
    (opts, args) = parser.parse_args()

    settings_list = get_settings_list(opts.setting_dir)

    batch_run_in_parallel(settings_list, int(opts.cores), verbose=opts.verbose)

