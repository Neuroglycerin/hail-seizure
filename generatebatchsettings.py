#!/usr/bin/env python3
import os
import json
import argparse


def get_default_settings():
    defaultsettings = {
        "R_SEED" : 5440,
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
        "VERSION": "_v2",
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
    featlist = [
                'feat_act',
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
                'feat_FFT',
                'feat_FFTcorrcoef',
                'feat_FFTcorrcoefeig',
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
                'feat_psd_logf',
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
    modlist = ['cln,raw,dwn','cln,ica,dwn','cln,csp,dwn']
    return modlist


def get_classifierlist():
    '''
    Provides a list of available classifiers.
    '''
    classifierlist = ['SVC','RandomForest','ExtraTrees','AdaBoost']
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
                      
    parser.add_argument("--aucinsamefolder",
                      action="store_true",
                      dest="aucinsamefolder",
                      default=False,
                      help="Save AUC predictions to the generated folder")
                      

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
                      
    parser.add_argument("-t", "--solomod",
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
    return parser
    
    
def parse_parser():
    parser = get_genbatch_parser()
    args = parser.parse_args()
    
    # Check inputs are okay
    if args.numcombined!=1:
        raise ValueError("Number combined is not 1")
    
    # Add more complex default inputs
    if args.doallfeatures:
        args.featurenames = get_featlist()
    if args.modtyps==[] or args.modtyps==None:
        args.modtyps = get_modlist()
    if args.doallclassifiers:
        args.classifiers = get_classifierlist()
    
    return args
    
    
def print_verbose(string, flag=False):
    '''
    Print statement only if flag is true
    '''
    if type(flag) is not bool:
        raise ValueError("verbose flag is not bool")
    if flag:
        print(string)
        

def write_settingsjson(settings, args):
    '''
    Writes a set of parameters as a .json file
    '''
    for classifier in args.classifiers:
        settings["CLASSIFIER"] = classifier
        if classifier=='SVC':
            shortclassifier = 'SVC'
        elif classifier=='RandomForest':
            shortclassifier = 'RF'
        elif classifier=='ExtraTrees':
            shortclassifier = 'XT'
        elif classifier=='AdaBoost':
            shortclassifier = 'AB'
        else:
            shortclassifier = classifier
        # Note if we are not using pseudodata
        if args.nopseudo:
            shortclassifier = shortclassifier + '_np'
            
        for split in args.numdatasplits:
        
            for modtyp in args.modtyps:
                if modtyp=='raw':
                    modtyp = 'cln,raw,dwn'
                elif modtyp=='ica':
                    modtyp = 'cln,ica,dwn'
                elif modtyp=='csp':
                    modtyp = 'cln,csp,dwn'
                
                
                if modtyp=='cln,raw,dwn':
                    shortmodtyp = 'raw'
                elif modtyp=='cln,ica,dwn':
                    shortmodtyp = 'ica'
                elif modtyp=='cln,csp,dwn':
                    shortmodtyp = 'csp'
                else:
                    shortmodtyp = modtyp
                
                if modtyp=='dirtyraw':
                    modtyp = 'raw'
                elif modtyp=='dirtyica':
                    modtyp = 'ica'
                elif modtyp=='dirtycsp':
                    modtyp = 'csp'
                    
                for feature in args.featurenames:
                    if split==1:
                        settings["FEATURES"] = ['{0}_{1}_'.format(modtyp, feature)]
                    else:
                        settings["FEATURES"] = ['{0}_{2}{1}_'.format(modtyp, feature, split)]
                    fname = '{0}_{1}_{2}.json'.format(shortclassifier, shortmodtyp, feature[5:])
                    
                    with open(args.outputdir+'/'+fname, 'w') as outfile:
                        json.dump(settings, outfile)


def main():
    args = parse_parser()
    settings = get_default_settings()
    if args.nopseudo:
        settings["DATA_TYPES"] = ["interictal","preictal","test"]
    else:
        settings["DATA_TYPES"] = ["interictal","preictal","test","pseudointerictal","pseudopreictal"]
    settings["CVITERCOUNT"] = args.numcvruns
    
    if args.aucinsamefolder:
        settings["AUC_SCORE_PATH"] = args.outputdir
    
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    
    write_settingsjson(settings, args)
    
    
if __name__=='__main__':
    main()
    

