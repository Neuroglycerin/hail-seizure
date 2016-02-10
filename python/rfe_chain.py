
import json
import train
import random
import os
from generatebatchsettings import *
import numpy as np
import copy
import hashlib
import glob
import sys
import optparse

import utils


def main(run_dir="rfe_chain", start=None, start_auc=None,
         verbose=None, logfile=None):
    """
    Main function to run the chain.
    """
    if logfile is not None:
        sys.stdout = open(logfile, "w")

    # load starting json
    with open(start) as f:
        start = json.load(f)
    if start_auc is None:
        startauc = 0.8

    start['AUC_SCORE_PATH'] = run_dir

    # have to load a list of possible features to replace with
    if all("10feat" in feature for feature in start['FEATURES']):
        with open("10featlist.json") as fh:
            featlist = json.load(fh)['FEATURES']
    else:
        featlist = get_featlist()

    # and possible preceding modifiers
    modlist = get_modlist()

    # creat list of combinations of these two lists
    comblist = []
    for mod in modlist:
        for feature in featlist:
            comblist.append('{0}_{1}_'.format(mod, feature))

    # define sampled json
    prevsample = copy.deepcopy(start)

    # initialise auc
    prevauc = startauc

    first = 1
    counter = 0
    converged = False
    # will decide what constitutes converged later
    while not converged:

        sample = copy.deepcopy(prevsample)
        # If this isn't the first one, sample new settings
        if not first:
            # Sample a new hdf5 and replace existing at random
            #   Or, just push it in, or just drop a hdf5 at random
            utils.print_verbose("===== Sampling new proposal "
                                "settings ======", flag=verbose)
            # sample new settings
            # shuffle combinations
            random.shuffle(comblist)

            # pop 3 features off this
            added = [comblist.pop() for i in range(3)]

            # add them to the settings
            sample['FEATURES'] = added

            utils.print_verbose("============================"
                                "===============", flag=verbose)

        # ensure that ordering of the features is the same between jsons
        sample['FEATURES'].sort()

        # Then save this new json with a descriptive name
        # unless it's already been generated
        if first:
            featurerecord = "".join(sample['FEATURES'])
        else:
            featurerecord = featurerecord + "".join(sample['FEATURES'])
        md5name = hashlib.md5(featurerecord.encode('UTF-8')).hexdigest()
        # get a list of the files in the run_dir
        existingjsons = glob.glob(run_dir + "/*.json")
        # check if the md5 exists
        if md5name + ".json" in existingjsons:
            # then load the results of that run
            with open(os.path.join(run_dir, "AUC_scores.csv"), "r") as fh:
                c = csv.reader(fh, delimiter="\t")
                utils.print_verbose("Already ran {0},"
                                    "reading from results.".format(md5name), flag=verbose)
                for line in c:
                    # look for that md5sum
                    if md5name in line[0]:
                        auc_score = line[-1]
        else:
            # save a json with this name and run train.py on it
            samplefname = os.path.join(run_dir, md5name + ".json")
            utils.print_verbose("Creating new settings"
                                " file for {0}".format(samplefname), flag=verbose)
            with open(samplefname, "w") as fh:
                json.dump(sample, fh)
            # call train.py
            try:
                if first:
                    auc_score_dict = train.main(samplefname, verbose=verbose,
                                                store_models=False, store_features=True)
                else:
                    picklefname = prevsamplefname.split(".")[0] + \
                        "_feature_dump.pickle"
                    # load the features saved in the last run
                    auc_score_dict = train.main(samplefname, verbose=verbose,
                                                store_models=False, store_features=True,
                                                load_pickled=picklefname)
                prevsamplefname = samplefname
                auc_score = auc_score_dict['all']
            except IndexError:
                print("Warning: accidentally added invalid feature.")
                os.remove(samplefname)
                # set auc to zero so these settings are not accepted
                auc_score = 0

        prevsample = sample

        # can't be first anymore
        first = 0

        # as it may be bad manners to run infinite loops
        counter += 1
        if counter > 100:
            converged = True

    return None


def get_parser():
    '''
    Generate parser for cmdline options.
    '''
    parser = optparse.OptionParser()

    parser.add_option("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Print verbose output")

    parser.add_option("-s", "--start",
                      action="store",
                      dest="start",
                      default="startchain_gavin.json",
                      help="Settings file to start at in JSON format"
                      "(default=startchain_gavin.json)")

    parser.add_option("-a", "--start_auc",
                      action="store",
                      dest="start_auc",
                      default=None,
                      help="AUC score of settings file starting the chain at"
                      "(default=None)")

    parser.add_option("-l", "--log",
                      action="store",
                      dest="log",
                      default=None,
                      help="Log file for verbose output of script"
                      "(default=None)")

    parser.add_option("-d", "--dir",
                      action="store",
                      dest="dir",
                      default="rfe_chain",
                      help="Directory to store jsons "
                      "(default=rfe_chain)")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    (opts, args) = parser.parse_args()
    main(
        run_dir=opts.dir,
        start=opts.start,
     start_auc=opts.start_auc,
     verbose=opts.verbose,
     logfile=opts.log)
