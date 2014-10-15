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
    subjects = settings['SUBJECTS']
    types = settings['DATA_TYPES']

    # open h5 read-only file for correct subj and version number

    h5_file_name = "{0}/{1}{2}.h5".format(feature_location, feat, version)
    try:
        h5_from_matlab = h5py.File(h5_file_name, 'r')
    except:
        assert False, "File: {0} does not exist".format(h5_file_name)

    # parse h5 object into dict (see docstring for struct)

    feature_dict = {}

    for subj in subjects:
        #loop through subjects and initialise the outer subj dict
        feature_dict.update({subj: {}})
        for typ in types:
            #loop through desired types and initialise typ dict for each subj
            feature_dict[subj].update({typ: {}})

            #because not all of next level have multiple values need
            #need to check whether it is a list of segs or just a value
            dataformat = type(h5_from_matlab[subj][typ])
            if dataformat is h5py._hl.group.Group:
                #if it is a list of segments just iterate over them and add to dict
                for seg in h5_from_matlab[subj][typ]:
                    feature_dict[subj][typ].update({seg: h5_from_matlab[subj][typ][seg].value})
            elif dataformat is h5py._hl.dataset.Dataset:
                #if it isn't a list of segements just add value directly under the typ dict
                    feature_dict[subj][typ]=h5_from_matlab[subj][typ].value

    # make sure h5 object is closed
    h5_from_matlab.close()

    return feature_dict

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

def build_training(subject, features, data):
    '''
    Build labelled data set for training
    input : subject  (subject name string)
            features (features to use)
    output: X (feature matrix as np.array)
            y (target vector as np.array)

    Does not preserve feature names
    '''

    # hacking this for later
    first = features[0]
    for feature in features:
        Xf = np.array([])

        # enumerate to get numbers for target vector:
        #     0 is interictal
        #     1 is preictal
        for i,ictal in enumerate(['interictal','preictal']):
            for segment in data[feature][subject][ictal].keys():
                # now stack up the feature vectors
                try:
                    Xf = np.vstack([Xf,
                                    np.ndarray.flatten(data[feature][subject][ictal][segment].T)])
                except ValueError:
                    Xf = np.ndarray.flatten(data[feature][subject][ictal][segment].T)
                # and stack up the target vector
                # but only for the first feature (will be the same for the rest)
                if feature == first:
                    try:
                        y.append(i)
                    except NameError:
                        y = [i]
        # stick the X arrays together
        try:
            X = np.hstack([X, Xf])
        except NameError:
            X = Xf
        except ValueError:
            print(feature)
            print(X.shape, Xf.shape)

    # turn y into an array
    y = np.array(y)
    return X, y

def build_test(subject, features, data):
    '''
    Function to build data structures for submission
    input : subject  (subject name string)
            features (features to use)
            data (nested dicts datastructure of all feature extracted data)
    output: X (feature matrix as np.array)
            y (labels as np.array)
    '''

    Xd = {}
    for feature in features:
        for segment in data[feature][subject]['test'].keys():
            fvector = np.ndarray.flatten(data[feature][subject]['test'][segment])
            try:
                Xd[segment] = np.hstack([Xd[segment], fvector])
            except:
                Xd[segment] = fvector

    # make the X array and corresponding labels
    segments = []
    X = []
    for segment in Xd.keys():
        segments.append(segment)
        X.append(Xd[segment])
    X = np.vstack(X)
    return X, segments

def output_csv(prediction_dict, settings=json_settings):
    '''
    Parse the predictions and output them in the correct format
    for submission to the output directory
    input:  prediction_dict (dictionary of all predictions of the test data)
            settings (the settings dict from parsing the json_object)
    output: void
    '''
    output_file = '{0}/output_{1}.csv'.format(settings['SUBMISSION_PATH'],
                                              settings['VERSION'])
    with open(output_file, 'w') as output_fh:
        csv_output = csv.writer(output_fh)
        csv_output.writerow(['clip', 'preictal'])
        for segment in prediction_dict.keys():
            csv_output.writerow.keys([segment, str(prediction_dict[segment])])
