import json
import numpy as np
import os
import h5py
from sklearn.externals import joblib #pickle w/ optimisation for np arrays

with open('SETTINGS.json') as settings_fh:
    json_settings = json.load(settings_fh)

def parse_matlab_HDF5(subj, settings=json_settings):
    '''
    Parse h5 file from matlab into hierarchial dict containing np arrays
    input: subj - subject name (e.g. Dog1),
           settings - parsed json settings in dict format (json.load)
    output: subject_dict containing data in hierarchial format
                    e.g. Dog_1 = {'interictal':
                                        {'feat1': featvector (np.array),
                                         'feat2 : featvector (np.array),
                                         ...},
                                  'preictal':
                                        {'feat1': featvector (np.array),
                                         ...},
                                   'test': ...}
    '''


    feature_location = settings['TRAIN_DATA_PATH']
    version = settings['VERSION']

    # open h5 read-only file for correct subj and version number
    h5_file_name = "{0}/{1}{2}.h5".format(feature_location, subj, version)
    print h5_file_name
    print feature_location
    print subj
    print version
    h5_from_matlab = h5py.File(h5_file_name, 'r')

    # parse h5 object into dict using nested comprehensions (see docstring
    # for struct)
    subject_dict = {typ: {feature: h5_from_matlab[typ][feature].value for feature in h5_from_matlab[typ]} for typ in h5_from_matlab}

    # make sure h5 object is closed
    h5_from_matlab.close()

    return subject_dict

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
