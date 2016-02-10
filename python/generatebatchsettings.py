#!/usr/bin/env python3
import os
import json
import argparse
import itertools
import numpy as np


def get_default_settings():
    defaultsettings = {
        "R_SEED": 5440,
        "AUC_SCORE_PATH": "auc_scores",
        "TRAIN_DATA_PATH": "train",
        "SUBJECTS": ["Dog_1",
                     "Dog_2",
                     "Dog_3",
                     "Dog_4",
                     "Dog_5",
                     "Patient_1",
                     "Patient_2"],
        "TEST_DATA_PATH": "test",
        "SUBMISSION_PATH": "output",
        "VERSION": "_v3",
        "RAW_DATA_DIRS": ["/disk/data2/neuroglycerin/hail-seizure-data/",
                          "/disk/data1/neuroglycerin/hail-seizure-data/",
                          "/disk/scratch/s1145806/hail-seizure-data/",
                          "/media/SPARROWHAWK/neuroglycerin/hail-seizure-data/",
                          "/media/scott/SPARROWHAWK/neuroglycerin/hail-seizure-data/"]
    }
    return defaultsettings


def get_featlist():
    '''
    Provides a list of all feature names.
    '''
    featlist = ['feat_act',
                'feat_ampcorrcoef-alpha',
                'feat_ampcorrcoef-alpha-eig',
                'feat_ampcorrcoef-beta',
                'feat_ampcorrcoef-beta-eig',
                'feat_ampcorrcoef-delta',
                'feat_ampcorrcoef-delta-eig',
                'feat_ampcorrcoef-high_gamma',
                'feat_ampcorrcoef-high_gamma-eig',
                'feat_ampcorrcoef-low_gamma',
                'feat_ampcorrcoef-low_gamma-eig',
                'feat_ampcorrcoef-theta',
                'feat_ampcorrcoef-theta-eig',
                'feat_coher_logf',
                'feat_corrcoef',
                'feat_corrcoefeig',
                'feat_cov',
                'feat_emvar-ARF',
                'feat_emvar-eCOHphs',
                'feat_emvar-eGphs',
                'feat_emvar-eSphs',
                'feat_emvar-PDCphs',
                'feat_emvar-COHphs',
                'feat_emvar-eCOH',
                'feat_emvar-eG',
                'feat_emvar-eS',
                'feat_emvar-PDC',
                'feat_emvar-COH',
                'feat_emvar-eDCphs',
                'feat_emvar-ePCOHphs',
                'feat_emvar-GPDCphs',
                'feat_emvar-Pphs',
                'feat_emvar-DCphs',
                'feat_emvar-eDC',
                'feat_emvar-ePCOH',
                'feat_emvar-GPDC',
                'feat_emvar-P',
                'feat_emvar-DC',
                'feat_emvar-eDDCphs',
                'feat_emvar-ePDCphs',
                'feat_emvar-Hphs',
                'feat_emvar-Sphs',
                'feat_emvar-DTFphs',
                'feat_emvar-eDDC',
                'feat_emvar-ePDC',
                'feat_emvar-H',
                'feat_emvar-S',
                'feat_emvar-DTF',
                'feat_emvar-eDPDCphs',
                'feat_emvar-ePphs',
                'feat_emvar-PCOHphs',
                'feat_emvar-eARF',
                'feat_emvar-eDPDC',
                'feat_emvar-eP',
                'feat_emvar-PCOH',
                'feat_FFT',
                'feat_FFTcorrcoef',
                'feat_FFTcorrcoefeig',
                'feat_gcaus',
                'feat_ilingam-causalindex',
                'feat_ilingam-causalorder',
                'feat_ilingam-connweights',
                'feat_lmom-1',
                'feat_lmom-2',
                'feat_lmom-3',
                'feat_lmom-4',
                'feat_lmom-5',
                'feat_lmom-6',
                'feat_mvar-ARF',
                'feat_mvar-COH',
                'feat_mvar-COHphs',
                'feat_mvar-DC',
                'feat_mvar-DCphs',
                'feat_mvar-DTF',
                'feat_mvar-DTFphs',
                'feat_mvar-GPDC',
                'feat_mvar-GPDCphs',
                'feat_mvar-H',
                'feat_mvar-Hphs',
                'feat_mvar-P',
                'feat_mvar-PCOH',
                'feat_mvar-PCOHphs',
                'feat_mvar-PDC',
                'feat_mvar-PDCphs',
                'feat_mvar-Pphs',
                'feat_mvar-S',
                'feat_mvar-Sphs',
                'feat_phase-alpha-dif',
                'feat_phase-alpha-sync',
                'feat_phase-beta-dif',
                'feat_phase-beta-sync',
                'feat_phase-delta-dif',
                'feat_phase-delta-sync',
                'feat_phase-high_gamma-dif',
                'feat_phase-high_gamma-sync',
                'feat_phase-low_gamma-dif',
                'feat_phase-low_gamma-sync',
                'feat_phase-theta-dif',
                'feat_phase-theta-sync',
                'feat_pib',
                'feat_pib_ratio',
                'feat_pib_ratioBB',
                'feat_PSDcorrcoef',
                'feat_PSDcorrcoefeig',
                'feat_psd',
                'feat_psd_logf',
                'feat_psd_logfBB',
                'feat_PSDlogfcorrcoef',
                'feat_PSDlogfcorrcoefeig',
                'feat_pwling1',
                'feat_pwling2',
                'feat_pwling4',
                'feat_pwling5',
                'feat_spearman',
                'feat_var',
                'feat_xcorr-tpeak',
                'feat_xcorr-twidth',
                'feat_xcorr-ypeak',
    ]
    return featlist


def get_modlist():
    '''
    Provides a list of all proprocessing models which can be used.
    '''
    modlist = ['cln,raw,dwn', 'cln,ica,dwn', 'cln,csp,dwn']
    return modlist


def get_classifierlist():
    '''
    Provides a list of available classifiers.
    '''
    classifierlist = ['SVC', 'RandomForest', 'ExtraTrees', 'AdaBoost']
    return classifierlist


def get_genbatch_parser():
    '''
    Generate optparse parser object for batch_run_script.py
    with the relevant options
    input:  void
    output: optparse parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        dest="verbose",
                        default=False,
                        help="Print verbose output")

    parser.add_argument("-d", "--dir",
                        action="store",
                        dest="outputdir",
                        required=True,
                        help="Directory holding json settings files")

    groupout = parser.add_mutually_exclusive_group(required=False)

    groupout.add_argument("-g", "--globaloutput",
                          action="store_true",
                          dest="useglobaloutput",
                          default=False,
                          help="Save AUC predictions to the global AUC csv folder")

    groupout.add_argument("-o", "--aucoutput",
                          action="store",
                          dest="aucoutput",
                          default="",
                          help="Save AUC predictions to a specific folder")

    groupfeat = parser.add_mutually_exclusive_group(required=True)

    groupfeat.add_argument("-f", "--featurenames",
                           action="store",
                           dest="featurenames",
                           default=[],
                           nargs='+',
                           help="List of feature names")

    groupfeat.add_argument("-a", "--allfeatures",
                           action="store_true",
                           dest="doallfeatures",
                           default=False,
                           help="Use all featurenames")

    groupclass = parser.add_mutually_exclusive_group(required=False)

    groupclass.add_argument("-c", "--classifier",
                            action="store",
                            dest="classifiers",
                            default=["SVC"],
                            nargs='+',
                            help="List of classifiers")

    groupclass.add_argument("--allclassifiers",
                            action="store_true",
                            dest="doallclassifiers",
                            default=False,
                            help="Use all classifiers")

    parser.add_argument("-m", "--modtyps",
                        action="store",
                        dest="modtyps",
                        default=[],
                        nargs='+',
                        help="Directory holding json settings files")

    parser.add_argument("--nopseudo",
                        action="store_true",
                        dest="nopseudo",
                        default=False,
                        help="Number of features to use at once")

    parser.add_argument("--splits",
                        action="store",
                        dest="numdatasplits",
                        type=int,
                        default=[1],
                        nargs='+',
                        help="Number of data splits (1 or 10) to use")

    parser.add_argument("-n", "--numcombined",
                        action="store",
                        dest="numcombined",
                        type=int,
                        default=1,
                        help="Number of features to use at once")

    parser.add_argument("--solomod",
                        action="store_true",
                        dest="dosinglemod",
                        default=False,
                        help="Whether to combine features with different modtyps")

    parser.add_argument("-k", "--numcvruns",
                        action="store",
                        dest="numcvruns",
                        type=int,
                        default=10,
                        help="Number of times to run through the cross-validator")

    parser.add_argument("-s", "--selection",
                        action="store",
                        dest="selection",
                        default=[],
                        nargs='+',
                        help="Key value pair for selection method")

    parser.add_argument("-t", "--threshold",
                        action="store",
                        dest="threshold",
                        default=0,
                        help="Threshold for selection method")

    parser.add_argument("--pre",
                        action="store",
                        dest="prestr",
                        default='',
                        help="String to go in front of all JSON names")

    parser.add_argument("--post",
                        action="store",
                        dest="poststr",
                        default='',
                        help="String to go in at the end of all JSON names")

    return parser


def parse_parser():
    parser = get_genbatch_parser()
    args = parser.parse_args()

    # Add more complex default inputs

    # If we're doing all features, get the list
    if args.doallfeatures:
        args.featurenames = get_featlist()
    # If we're doing all modtyps, get the list
    if args.modtyps == [] or args.modtyps is None:
        args.modtyps = get_modlist()
    # If we're doing all classifiers, get the list
    if args.doallclassifiers:
        args.classifiers = get_classifierlist()
    # Use the global AUC output csv if requested
    if args.useglobaloutput:
        args.aucoutput = "auc_scores"
    # By default, we will output in a CSV in the same folder as the batch
    # folder
    if args.aucoutput == "":
        args.aucoutput = args.outputdir

    return args


def print_verbose(string, flag=False):
    '''
    Print statement only if flag is true
    '''
    if not isinstance(flag, bool):
        raise ValueError("verbose flag is not bool")
    if flag:
        print(string)


def write_settingsjson(settings, args):
    '''
    Writes a set of parameters as a .json file
    '''
    for classifier in args.classifiers:
        settings["CLASSIFIER"] = classifier
        # Shorten the classifier name, to save filename space
        if classifier == 'SVC':
            shortclassifier = 'SVC'
        elif classifier == 'RandomForest':
            shortclassifier = 'RF'
        elif classifier == 'ExtraTrees':
            shortclassifier = 'XT'
        elif classifier == 'AdaBoost':
            shortclassifier = 'AB'
        else:
            shortclassifier = classifier

        # Record if we are not using pseudodata
        # Add it to the classifier savename
        if args.nopseudo:
            shortclassifier = shortclassifier + '_np'

        for split in args.numdatasplits:

            fullfeatstrlst = []
            shortfeatstrlst = []

            for iMod, modtyp in enumerate(args.modtyps):
                # Use these as shorthands for the clean versions
                if modtyp == 'raw':
                    modtyp = 'cln,raw,dwn'
                elif modtyp == 'ica':
                    modtyp = 'cln,ica,dwn'
                elif modtyp == 'csp':
                    modtyp = 'cln,csp,dwn'

                # Save the clean versions without full name to save space
                if modtyp == 'cln,raw,dwn':
                    shortmodtyp = 'raw'
                elif modtyp == 'cln,ica,dwn':
                    shortmodtyp = 'ica'
                elif modtyp == 'cln,csp,dwn':
                    shortmodtyp = 'csp'
                else:
                    shortmodtyp = modtyp

                # But still allow to use the dirty versions on request
                if modtyp == 'dirtyraw':
                    modtyp = 'raw'
                elif modtyp == 'dirtyica':
                    modtyp = 'ica'
                elif modtyp == 'dirtycsp':
                    modtyp = 'csp'

                if not modtyp == '':
                    mymod = '_' + modtyp
                else:
                    mymod = modtyp
                myfull = []
                myshort = []

                # Make a list of all features with this modtyp
                for iFtr, feature in enumerate(args.featurenames):
                    if feature[-3:] == '.h5':
                        feature = feature[:-6]
                    if feature[-1:] == '_':
                        feature = feature[:-1]
                    # Have to have a special case for the unsplit segments
                    if split == 1:
                        myfull.append('{0}{1}_'.format(mymod, feature))
                    else:
                        myfull.append(
                            '{0}{2}{1}_'.format(
                                mymod,
                                feature,
                                split))
                    # Check if starts with feat_
                    if feature[0:5] == 'feat_':
                        # The short version does not need "feat_" at beginning
                        myfsh = feature[5:]
                    else:
                        # This is a weird feature function...
                        myfsh = feature
                    myshort.append('{0}_{1}'.format(shortmodtyp, myfsh))

                fullfeatstrlst.append(myfull)
                shortfeatstrlst.append(myshort)

            if not args.dosinglemod:
                fullfeatstrlst = np.array(fullfeatstrlst).flatten()
                fullfeatstrlst = [fullfeatstrlst]
                shortfeatstrlst = np.array(shortfeatstrlst).flatten()
                shortfeatstrlst = [shortfeatstrlst]

            # Loop over every modtyp
            for iMod in range(len(fullfeatstrlst)):

                # Make a combinatorial combination of features
                for i in itertools.combinations(range(len(fullfeatstrlst[iMod])), args.numcombined):

                    myfeats = []
                    myshortfeats = []

                    # Add together each feature in this combination
                    for j in range(args.numcombined):
                        myfeats.append(fullfeatstrlst[iMod][i[j]])
                        myshortfeats.append(shortfeatstrlst[iMod][i[j]])

                    settings["FEATURES"] = myfeats

                    ff = '_AND_'.join(myshortfeats)
                    fname = '{2}{0}_{1}{3}.json'.format(
                        shortclassifier,
                        ff,
                        args.prestr,
                        args.poststr)

                    # Output to a JSON
                    with open(args.outputdir + '/' + fname, 'w') as outfile:
                        json.dump(settings, outfile)


def main():
    args = parse_parser()
    settings = get_default_settings()
    if args.nopseudo:
        settings["DATA_TYPES"] = ["interictal", "preictal", "test"]
    else:
        settings["DATA_TYPES"] = ["interictal",
                                  "preictal",
                                  "test",
                                  "pseudointerictal",
                                  "pseudopreictal"]
    settings["CVITERCOUNT"] = args.numcvruns

    if len(args.selection) == 1:
        settings["SELECTION"] = {args.selection[0]: None}
    elif len(args.selection) == 2:
        settings["SELECTION"] = {args.selection[0]: int(args.selection[1])}
    elif not len(args.selection) == 0:
        print(
            'Error incorrect number of selection inputs: {0}'.format(len(args.selection)))

    settings["THRESHOLD"] = args.threshold

    settings["AUC_SCORE_PATH"] = args.outputdir
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    write_settingsjson(settings, args)


if __name__ == '__main__':
    main()
