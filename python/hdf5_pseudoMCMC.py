# coding: utf-8

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
import discriminate

import utils


def main(mcmcdir="hdf5mcmc", start=None, start_auc=None,
         verbose=True, logfile=None, discr_flag=False):
    """
    Contains the main loop for this script.
    Pseudo-MHMCMC to find optimal AUC scoring
    combinations of HDF5s.
    start - location of json file settings to begin at
    """
    if logfile is not None:
        sys.stdout = open(logfile, "w")
    # pseudo-code for the MCMC iteration
    # want it to start with the probably good features
    with open(start) as f:
        start = json.load(f)
    if start_auc is None:
        startauc = 0.8

    # hardcode AUC results to the hdf5mcmc directory
    start['AUC_SCORE_PATH'] = mcmcdir

    # have to load a list of possible features to replace with
    if all("10feat" in feature for feature in start['FEATURES']):
        with open("10featlist.json") as fh:
            featlist = json.load(fh)['FEATURES']
    else:
        featlist = get_featlist()

    # and possible preceding modifiers
    modlist = get_modlist()

    # define sampled json
    prevsample = copy.deepcopy(start)

    # initialise auc
    prevauc = startauc

    counter = 0
    converged = False
    # will decide what constitutes converged later
    while not converged:

        sample = copy.deepcopy(prevsample)
        # Sample a new hdf5 and replace existing at random
        #   Or, just push it in, or just drop a hdf5 at random
        utils.print_verbose("===== Sampling new proposal "
                            "settings ======", flag=verbose)
        u = np.random.rand()
        if u < 0.25:
            # drop an element at random
            features = sample['FEATURES'][:]
            random.shuffle(features)
            dropped = features.pop()
            sample['FEATURES'] = features
            utils.print_verbose(
                "Dropped feature {0}".format(dropped),
                flag=verbose)
        elif u > 0.25 and u < 0.5:
            # keep trying to sample a new feature until we
            # find one that's not in there already
            while True:
                # push a new feature, but don't remove an old one
                newfeature = random.sample(featlist, 1)[0]
                newmod = random.sample(modlist, 1)[0]
                added = '{0}_{1}_'.format(newmod, newfeature)
                if added not in sample['FEATURES']:
                    break
            sample['FEATURES'].append(added)
            utils.print_verbose(
                "Added feature {0}".format(added),
                flag=verbose)
        elif u > 0.5:
            # push a new feature and remove an old one
            features = sample['FEATURES'][:]
            random.shuffle(features)
            dropped = features.pop()
            # keep trying to sample a new feature until we
            # find one that's not in there already
            while True:
                # push a new feature, but don't remove an old one
                newfeature = random.sample(featlist, 1)[0]
                newmod = random.sample(modlist, 1)[0]
                added = '{0}_{1}_'.format(newmod, newfeature)
                if added not in sample['FEATURES']:
                    break
            features.append(added)
            sample['FEATURES'] = features
            utils.print_verbose("Switched feature {0} for "
                                "{1}".format(dropped, added), flag=verbose)
        utils.print_verbose("============================"
                            "===============", flag=verbose)
        # ensure that ordering of the features is the same between jsons
        sample['FEATURES'].sort()

        # Then save this new json with a descriptive name
        # unless it's already been generated
        md5name = hashlib.md5(
            "".join(sample['FEATURES']).encode('UTF-8')).hexdigest()
        # get a list of the files in the mcmcdir
        existingjsons = glob.glob(mcmcdir + "/*.json")
        # check if the md5 exists
        if md5name + ".json" in existingjsons:
            # then load the results of that run
            with open(os.path.join(mcmcdir, "AUC_scores.csv"), "r") as fh:
                c = csv.reader(fh, delimiter="\t")
                utils.print_verbose("Already ran {0},"
                                    "reading from results.".format(md5name), flag=verbose)
                for line in c:
                    # look for that md5sum
                    if md5name in line[0]:
                        auc_score = line[-1]
        else:
            # save a json with this name and run train.py on it
            samplefname = os.path.join(mcmcdir, md5name + ".json")
            utils.print_verbose("Creating new settings"
                                " file for {0}".format(samplefname), flag=verbose)
            with open(samplefname, "w") as fh:
                json.dump(sample, fh)
            # call train.py or discriminate.py
            if discr_flag:
                try:
                    auc_score_dict = discriminate.main(samplefname,
                                                       verbose=verbose)
                    # don't want to rename this variable
                    # even though it is no longer an AUC score
                    # want a low accuracy score, strangely enough
                    auc_score = 1 - auc_score_dict['all']
                except IndexError:
                    print("Warning: accidentally added invalid feature.")
                    os.remove(samplefname)
                    # set auc to zero so these settings are not accepted
                    auc_score = 0
            else:
                try:
                    auc_score_dict = train.main(samplefname,
                                                verbose=verbose, store_models=False)
                    auc_score = auc_score_dict['all'] - 0.5
                except IndexError:
                    print("Warning: accidentally added invalid feature.")
                    os.remove(samplefname)
                    # set auc to zero so these settings are not accepted
                    auc_score = 0

        utils.print_verbose("==== Acceptance calculation ====", flag=verbose)
        # compute acceptance probability from AUC:
        #     r = min(1,AUC/(previous AUC))
        acceptance = np.max([np.min([1, auc_score / prevauc]), 0])

        u = np.random.rand()
        # accept new point with probability r
        if u < acceptance:
            prevsample = sample
            # save current auc
            prevauc = auc_score
            utils.print_verbose("accepting new settings with probability "
                                "{0}".format(acceptance), flag=verbose)
        else:
            utils.print_verbose("rejecting new settings with probability "
                                "{0}".format(1.0 - acceptance), flag=verbose)
        utils.print_verbose("================================", flag=verbose)
        # otherwise it will not overwrite prevsample, so continue from where it
        # was

        # as it may be bad manners to run infinite loops
        counter += 1
        if counter > 100:
            converged = True


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
                      default="hdf5mcmc",
                      help="Directory to store jsons "
                      "(default=hdf5mcmc)")

    parser.add_option("-D", "--disciminate",
                      action="store_true",
                      dest="discriminate",
                      default=False,
                      help="Instead of training the model using train.py"
                      " use discrimate.py.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    (opts, args) = parser.parse_args()
    main(
        mcmcdir=opts.dir,
        start=opts.start,
     start_auc=opts.start_auc,
     verbose=opts.verbose,
     logfile=opts.log,
     discr_flag=opts.discriminate)
