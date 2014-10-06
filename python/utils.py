import json
import numpy as np
import os
import h5py
from sklearn.externals import joblib #pickle w/ optimisation for np arrays

with open('SETTINGS.json') as settings_fh:
    json_settings = json.load(settings_fh)

def parse_matlab_HDF5(feat, settings=json_settings):
    '''
    Parse h5 file from matlab into hierarchial dict containing np arrays
    input: feat - feature name (e.g. Dog_1),
           settings - parsed json settings in dict format (json.load)
    output: feat_dict containing data in hierarchial format
                    e.g. raw_feat_cov= {'Dog_1': {
                                            'interictal': {segment_fname1: featvector,
                                                           segment_fname2: featvector}

                                             'preictal': { ... },
                                             'test': { ... }
                                             }
                                        'Dog_2': { ... { ... } }}
                                  }
    '''


    feature_location = settings['TRAIN_DATA_PATH']
    version = settings['VERSION']

    # open h5 read-only file for correct subj and version number

    h5_file_name = "{0}/{1}{2}.h5".format(feature_location, feat, version)
    h5_from_matlab = h5py.File(h5_file_name, 'r')

    # parse h5 object into dict using nested comprehensions (see docstring
    # for struct)
    feature_dict = {subj:
                        {typ:
                            {segment: h5_from_matlab[subj][typ][segment].value
                             for segment in h5_from_matlab[subj][typ]}
                        for typ in h5_from_matlab[subj]}
                    for subj in h5_from_matlab}

    # make sure h5 object is closed
    h5_from_matlab.close()

    return feature_dict

def reformat_features(subject_dict, typ):
    pass


def serialise_trained_model(model, model_name, settings=json_settings):
    '''
    Serialise and compress trained sklearn model to repo
    input: model (sklearn model)
           model_name (string for model file name)
           settings (parsed SETTINGS.json object)
    output: retcode
    '''
    joblib.dump(model, settings['MODEL_PATH']+'/'+model_name, compress=9)

def read_trained_model(model_name, settings=json_settings):
    '''
    Read trained model from repo
    input: model_name (string for model file name)
           settings (parsed SETTINGS.json object)
    output: model
    '''

    return joblib.load(settings['MODEL_PATH']+'/'+model_name)
