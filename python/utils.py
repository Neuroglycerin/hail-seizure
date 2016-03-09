import os
import sys
import argparse
import warnings
import pdb
import pickle
import csv
import json
import re
import math

import h5py
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.pipeline
import sklearn.externals


def get_parser():
    '''
    Generate argparse parser object for train and predict.py
    with the relevant options
    input:  void
    output: argparse parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print verbose output",
    )
    parser.add_argument(
        "-s", "--settings",
        action="store",
        dest="settings",
        default="SETTINGS.json",
        help="Settings file to use in JSON format (default=SETTINGS.json)",
    )
    parser.add_argument(
        "-j", "--cores",
        action="store",
        dest="parallel",
        default=0,
        help="Train subjects in parallel and with specified num of cores",
    )
    parser.add_argument(
        "-p", "--pickle",
        action="store",
        dest="pickle_detailed",
        default=False,
        help="Pickle file to save detailed training results for further"
             " analysis (default=False)",
    )
    return parser


def get_settings(settings_file):
    '''
    Small wrapped for json.load to parse settings file
    input:  settings_file - string filename for settings file
    output: settings - dict containing parsed settings values
    '''
    with open(settings_file, 'r') as sett_fh:
        settings = json.load(sett_fh)

    # add settings file name (basename) to settings dict
    # need to strip any additional path apart from basename
    settings_file_basename = os.path.basename(settings_file)

    # now strip off the extension (should be .json)
    settings_file_used = os.path.splitext(settings_file_basename)[0]

    settings.update({'RUN_NAME': settings_file_used})

    # make subjects, features, data_types immutable tuples

    for field in ['SUBJECTS', 'DATA_TYPES', 'FEATURES']:
        settings.update({field: tuple(settings[field])})

    # default directories for all paths
    default_directories = {'TRAIN_DATA_PATH': 'train',
                           'MODEL_PATH': 'model',
                           'SUBMISSION_PATH': 'output',
                           'AUC_SCORE_PATH': 'auc_scores'}

    if 'CVITERCOUNT' not in settings.keys():
        settings.update({'CVITERCOUNT': 10})

    # add missing default and update file paths settings to have full absolute paths
    for settings_field in default_directories.keys():
        if settings_field not in settings.keys():
            settings.update({settings_field: default_directories[settings_field]})

        settings[settings_field] = os.path.abspath(settings[settings_field])


    # reads the settings classifier field and return the appropriate clf obj
    # using the model mapping dict

    classifier_objs = {
        'RandomForest':
            sklearn.ensemble.RandomForestClassifier(
                random_state=settings['R_SEED'],
                ),
        'ExtraTrees':
            sklearn.ensemble.ExtraTreesClassifier(
                random_state=settings['R_SEED'],
                ),
        'AdaBoost':
            sklearn.ensemble.AdaBoostClassifier(
                random_state=settings['R_SEED'],
                ),
        'SVC':
            sklearn.svm.SVC(
                probability=True,
                random_state=settings['R_SEED'],
                ),
        'LinearSVC':
            # Use this instead of svm.LinearSVC even though it is
            # slower, because we use the sample_weights input
            sklearn.svm.SVC(
                random_state=settings['R_SEED'],
                kernel='linear',
                probability=True,
                ),
        'LogisticRegression':
            sklearn.linear_model.LogisticRegression(
                random_state=settings['R_SEED'],
                solver='sag',
                max_iter=5000,
                ),
        'RidgeClassifier':
            sklearn.linear_model.RidgeClassifier(
                random_state=settings['R_SEED'],
                ),
        'RandomTreesEmbedding':
            sklearn.ensemble.RandomTreesEmbedding(
                random_state=settings['R_SEED'],
                ),
        'GradientBoostingClassifier':
            sklearn.ensemble.RandomTreesEmbedding(
                random_state=settings['R_SEED'],
                ),
        'SGDClassifier':
            sklearn.ensemble.RandomTreesEmbedding(
                random_state=settings['R_SEED'],
                ),
    }

    # todo: bagging

    default_classifier = classifier_objs['SVC']

    # if there is no classifier specified or an unsupport of malformed settings
    # is provided then set classifier setting to default
    if 'CLASSIFIER' in settings.keys():
        try:
            settings['CLASSIFIER'] = classifier_objs[settings['CLASSIFIER']]
        except KeyError:
            warnings.warn("Classifier: {0} (specified in {1}), "
                          "using default instead: {2}".format(settings['CLASSIFIER'],
                                                          settings_file,
                                                          str(default_classifier)))
            settings.update({'CLASSIFIER': default_classifier})
    else:
        settings.update({'CLASSIFIER': default_classifier})

    # if there is a settings opt field convert
    if 'CLASSIFIER_OPTS' not in settings.keys():
        settings.update({'CLASSIFIER_OPTS': {}})


    # set the model with the params desired
    try:
        settings['CLASSIFIER'].set_params(**settings['CLASSIFIER_OPTS'])
    except ValueError:
        print("CLASSIFIER_OPTS {0} in {1} "
              "contains invalid parameters".format(settings_file,
                                                   str(settings['CLASSIFIER_OPTS'])))
        # exits with 0 if it fails as this will allow batch to complete and not fail
        sys.exit(0)

    return settings


def get_data(settings, verbose=False):
    '''
    Iterate through Feature HDF5s and parse input using
    parse_matlab_HDF5 into a dict
    input:  features - list of feature to parse
            settings - parsed settings file
    output: data - dict of {feature name: respective parsed HDF5}
    '''
    data = {}
    features = settings['FEATURES']
    for feat_name in features:
        print_verbose("** Parsing {0} **".format(feat_name), flag=verbose)
        parsed_feat = parse_matlab_HDF5(feat_name, settings)
        if parsed_feat is not None:
            data.update({feat_name: parsed_feat})
    return data


def print_verbose(string, flag=False):
    '''
    Print statement only if flag is true
    '''
    if not isinstance(flag, bool):
        raise ValueError("verbose flag is not bool")
    if flag:
        print(string)


def parse_matlab_HDF5(feat, settings):
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

    h5_file_name = os.path.join(feature_location, "{0}{1}.h5".format(\
            feat, version))

    # Try to open hdf5 file if it doesn't exist print error and return None
    try:
        h5_from_matlab = h5py.File(h5_file_name, 'r')
    except OSError:
        warnings.warn("{0} does not exist (or is not readable)"
                      "".format(h5_file_name))
        return None

    # parse h5 object into dict (see docstring for struct)

    feature_dict = {}
    try:
        for subj in subjects:
            # loop through subjects and initialise the outer subj dict
            feature_dict.update({subj: {}})

            for typ in types:
                # Not all HDF5s will have all types (e.g. CSP won't have MI etc)
                # Therefore check if type is present, continue next loop iter if
                # it isn't

                if typ in list(h5_from_matlab[subj]):
                    # loop through desired types and initialise typ dict
                    # for each subj
                    feature_dict[subj].update({typ: {}})

                    # Not all all of next level have multiple values so need
                    # need to check whether it is a list of segs or just a
                    # single value
                    dataformat = type(h5_from_matlab[subj][typ])

                    if dataformat is h5py._hl.group.Group:
                        # If it is a list of segments then just iterate
                        # over them and add to dict
                        for seg in h5_from_matlab[subj][typ]:
                            feature_dict[subj][typ].update(\
                                    {seg: h5_from_matlab[subj][typ][seg].value})

                    elif dataformat is h5py._hl.dataset.Dataset:
                        # if it isn't a list of segements just add value
                        # directly under the typ dict
                        feature_dict[subj][typ] = h5_from_matlab[subj][typ].value

                elif typ not in list(h5_from_matlab[subj]):
                    continue

    except:
        warnings.warn("Unable to parse {0}".format(h5_file_name))
        return None

    # make sure h5 object is closed
    h5_from_matlab.close()

    return feature_dict


def serialise_trained_model(model, subject, settings, verbose=False):
    '''
    Serialise and compress trained sklearn model to repo
    input: model (sklearn model)
           model_name (string for model file name)
           settings (parsed SETTINGS.json object)
    output: retcode
    '''
    model_name = "{0}_model_for_{1}_using_{2}_feats.model".format(\
                                                      settings['RUN_NAME'],
                                                      subject,
                                                      settings['VERSION'])

    print_verbose("##Writing Model: {0}##".format(model_name), flag=verbose)
    sklearn.externals.joblib.dump(model,
                os.path.join(settings['MODEL_PATH'], model_name),
                compress=9)


def read_trained_model(subject, settings, verbose=False):
    '''
    Read trained model from repo
    input: model_name (string for model file name)
           settings (parsed SETTINGS.json object)
    output: model
    '''
    model_name = "{0}_model_for_{1}_using_{2}_feats.model".format(\
                                                      settings['RUN_NAME'],
                                                      subject,
                                                      settings['VERSION'])

    print_verbose("##Loading Model: {0}##".format(model_name), flag=verbose)
    model = sklearn.externals.joblib.load(os.path.join(settings['MODEL_PATH'], model_name))

    return model


class DataAssembler:
    def __init__(self, settings, data, metadata):
        """
        A class to take the data (nested dictionaries) and intended features
        and produce training sets which can be used by scikit-learn.

        Initialisation:

        * settings - dictionary produced by json
        * metadata - dictionary produced by metadata json
        * data - data produced by get_data
        """
        # save initialisation to self
        self.metadata = metadata
        self.settings = settings
        self.data = data
        self.metadata = metadata

        # check if we're using minute long features
        if all('10feat' in feature for feature in settings['FEATURES']):
            self.minutefeatures = True
        elif any('10feat' in feature for feature in settings['FEATURES']) and not \
             all('10feat' in feature for feature in settings['FEATURES']):
            raise ValueError("Cannot mix 1 minute and 10 minute features.")
        else:
            self.minutefeatures = False

        # parse for segment tuple/list
        self.segments = self._parse_segment_names()

        # find out if we're dealing with pseudo data
        if 'pseudointerictal' in settings['DATA_TYPES'] and \
                'pseudopreictal' in settings['DATA_TYPES']:
            self.include_pseudo = True
        else:
            self.include_pseudo = False

        return None

    def _parse_segment_names(self):
        """
        Creates a dictionary of dictionaries, containing tuples of segment
        names:
        Output:
        * segment dictionaries {subject:{ictyp:[segment names, ...]}}
        """
        segments = {}
        for subject in self.settings['SUBJECTS']:
            segments[subject] = {}
            for ictyp in self.settings['DATA_TYPES']:
                segments[subject][ictyp] = []
        # get full list of possible segments
        all_segments = self._scrape_segments()

        # check flag for 1-minute segment features
        if self.minutefeatures:
            # there are 9 minute samples in pseudo segments
            # and 10 in regular segments
            self.minutemod = {'preictal': 10, 'interictal': 10,
                    'test': 10, 'pseudopreictal': 9, 'pseudointerictal': 9}
        else:
            self.minutemod = {'preictal': 1, 'interictal': 1,
                    'test': 1, 'pseudopreictal': 1, 'pseudointerictal': 1}

        # This will fix the order of the segments
        # iterate over all possible segments
        for segment in all_segments:
            # ensure test is specified as ictyp
            # and the subject is correct for test segments
            if 'test' in segment:
                ictyp = 'test'
                subject = [subject for subject in self.settings['SUBJECTS'] \
                          if subject in segment][0]

            else:
                segment_key = segment.split('.')[0]
                # for this segment, find what subject it's in
                subject = self.metadata[segment_key]['subject']
                # and what ictyp it is
                ictyp = self.metadata[segment_key]['ictyp']
                # store in the dictionary of dictionaries
            if ictyp in self.settings['DATA_TYPES']:
                segments[subject][ictyp] += [segment]*self.minutemod[ictyp]

        # then enforce tuple
        for subject in segments.keys():
            for ictyp in segments[subject].keys():
                segments[subject][ictyp] = tuple(segments[subject][ictyp])

        return segments

    def _scrape_segments(self):
        """
        Scrapes segment ids out of the data file.
        Ensures that each feature has the same segments covered.
        Output:
        * all_segments = total list of segments
        """
        # initialise set and fill with first feature's segments
        all_segments = set([])
        for subject in self.settings['SUBJECTS']:
            for ictyp in self.settings['DATA_TYPES']:
                all_segments |= set(
                    self.data[self.settings['FEATURES'][0]][subject][ictyp].keys())
        # iterate over all features to ensure that the segments are the same
        for feature in self.settings['FEATURES']:
            verification_segments = set([])
            for subject in self.settings['SUBJECTS']:
                for ictyp in self.settings['DATA_TYPES']:
                    verification_segments |= set(
                        self.data[feature][subject][ictyp].keys())
            if verification_segments != all_segments:
                raise ValueError("Feature {0} contains segments that "
                                 "do not match feature {1}.".format(feature,
                                        self.settings['FEATURES'][0]))
        # turn segments into a tuple
        all_segments = sorted(all_segments)
        all_segments = tuple(all_segments)
        return all_segments

    def _build_X(self, subject, ictyp):
        """
        Takes a subject string and ictal class string. Processes a
        feature vector matrix X corresponding to that subject.
        Also writes a vector of feature names corresponding to how
        they are arranged in the matrix.
        Input:
        * subject
        * ictyp
        Output:
        * X
        * feature_names
        """
        # iterate over features, calling _assemble feature
        # to build parts of the full X matrix
        X_parts = []
        feature_names = []
        for feature in self.settings['FEATURES']:
            X_part = self._assemble_feature(subject, feature, ictyp)
            X_parts += [X_part]
            # build vector of feature names,
            # with the length of the feature
            feature_names += [feature]*X_part.shape[1]
        # put these together with numpy
        X = np.hstack(X_parts)

        # assemble vector of feature names together
        # should be ['feature name',....]
        feature_names = np.hstack(feature_names)

        return X, feature_names

    def _assemble_feature(self, subject, feature, ictyp):
        """
        Create a matrix containing a feature vector in the order:
        Input:
        * feature - which feature to build the matrix of
        * subject
        * ictyp
        Output:
        * X_part - part of the X matrix
        """
        if self.minutefeatures:
            # initialise dictionary for features
            segment10feature = {}

        # iterate over segments and build the X_part matrix
        rows = []
        for segment in self.segments[subject][ictyp]:
            # check if the flag for 1-minute segments is set
            if self.minutefeatures:
                # if new segment initialise dictionary item
                if segment not in segment10feature:
                    featurearray = self.data[feature][subject][ictyp][segment]
                    # list of arrays for each minute
                    # how to slice depends on dimensionality

                    if len(featurearray.shape) == 2:
                        minute_segment_list = [featurearray[:, i] \
                                for i in range(self.minutemod[ictyp])]

                    elif len(featurearray.shape) == 3:
                        minute_segment_list = [featurearray[:,:, i] \
                                for i in range(self.minutemod[ictyp])]

                    elif len(featurearray.shape) == 4:
                        minute_segment_list = [featurearray[:,:,:, i] \
                                for i in range(self.minutemod[ictyp])]

                    else:
                        raise ValueError("Feature {0} has invalid number of"
                            " dimensions for a 1-minute feature".format(feature))
                    segment10feature[segment] = minute_segment_list
                    # pop an element off the list and flatten
                    row = np.ndarray.flatten(segment10feature[segment].pop())
                    rows += [row]
                # if not pop another array off the dictionary
                else:
                    row = np.ndarray.flatten(segment10feature[segment].pop())
                    rows += [row]
            else:
                # first flatten whatever is in the array returned
                row = np.ndarray.flatten(self.data[feature][subject][ictyp][segment])
                # gather up all the rows in the right order
                rows += [row]
        # stack up all the rows
        X_part = np.vstack(rows)

        return X_part

    def _build_y(self, subject, ictyp):
        """
        Takes a subject string and processes an feature vector
        matrix X corresponding to that subject.
        Input:
        * subject
        Output:
        * y
        """
        # check this is the right ictyp
        if ictyp == "test":
            raise ValueError
        y_length = 0
        # iterate over segments to count
        for segment in self.segments[subject][ictyp]:
            y_length += 1
        if 'preictal' == ictyp:
            return np.array([1]*y_length)
        elif 'interictal' == ictyp:
            return np.array([0]*y_length)
        elif 'pseudopreictal' == ictyp:
            return np.array([1]*y_length)
        elif 'pseudointerictal' == ictyp:
            return np.array([0]*y_length)
        else:
            raise ValueError

    def build_training(self, subject):
        """
        Builds a training set for a given subject.
        Input:
        * subject
        Output:
        * X,y
        """
        # for preictal and interictal call build y and build X
        # and stack them up
        verification_names = [[], [], []]
        X_inter, self.training_names = self._build_X(subject, 'interictal')
        X_pre, verification_names[0] = self._build_X(subject, 'preictal')
        if self.include_pseudo:
            X_psinter, verification_names[1] = self._build_X(subject,\
                    'pseudointerictal')
            X_pspre, verification_names[2] = self._build_X(subject,\
                    'pseudopreictal')

        if all(all(tr != vf for tr in self.training_names for \
                vf in verification) for verification in verification_names):
            raise ValueError
        if self.include_pseudo:
            X = np.vstack([X_inter, X_pre, X_psinter, X_pspre])
        else:
            X = np.vstack([X_inter, X_pre])
        if self.include_pseudo:
            y = np.hstack([self._build_y(subject, 'interictal'), \
                           self._build_y(subject, 'preictal'), \
                           self._build_y(subject, 'pseudointerictal'), \
                           self._build_y(subject, 'pseudopreictal')])
        else:
            y = np.hstack([self._build_y(subject, 'interictal'), \
                           self._build_y(subject, 'preictal')])
        # storing feature names in self.training_names

        # storing the correct sequence of segments
        if self.include_pseudo:
            self.training_segments = np.hstack([ \
                    np.array(self.segments[subject]['interictal']), \
                    np.array(self.segments[subject]['preictal']), \
                    np.array(self.segments[subject]['pseudointerictal']), \
                    np.array(self.segments[subject]['pseudopreictal'])   ])
        else:
            self.training_segments = np.hstack([ \
                    np.array(self.segments[subject]['interictal']), \
                    np.array(self.segments[subject]['preictal'])])

        return X, y


    def build_test(self, subject):
        """
        Builds test set for given subject.
        Input:
        * subject
        Output:
        * X
        """
        # storing names for the features in self.test_names
        X, self.test_names = self._build_X(subject, 'test')

        # storing the correct sequence of segments
        self.test_segments = np.array(self.segments[subject]['test'])

        return X

    def build_custom_training(self, subject, featurearray):
        """
        Takes a subject and array of names and feature indexes
        then builds a training set of precisely those feature elements,
        in that order.
        Input:
        * subject - string
        * namearray - array of structure [[feature_name, index],...]
        Output
        * X,y
        """
        # for preictal and interictal call build y and build X
        # and stack them up
        verification_names = [[], [], []]
        X_inter = self._build_custom_X(subject, 'interictal',\
                featurearray)
        X_pre = self._build_custom_X(subject, 'preictal',\
                featurearray)
        if self.include_pseudo:
            X_psinter = self._build_custom_X(subject,\
                    'pseudointerictal', featurearray)
            X_pspre = self._build_custom_X(subject,\
                    'pseudopreictal', featurearray)

        if self.include_pseudo:
            X = np.vstack([X_inter, X_pre, X_psinter, X_pspre])
        else:
            X = np.vstack([X_inter, X_pre])
        if self.include_pseudo:
            y = np.hstack([self._build_y(subject, 'interictal'), \
                           self._build_y(subject, 'preictal'), \
                           self._build_y(subject, 'pseudointerictal'), \
                           self._build_y(subject, 'pseudopreictal')])
        else:
            y = np.hstack([self._build_y(subject, 'interictal'), \
                           self._build_y(subject, 'preictal')])
        # storing feature names in self.training_names

        # storing the correct sequence of segments
        if self.include_pseudo:
            self.training_segments = np.hstack([ \
                    np.array(self.segments[subject]['interictal']), \
                    np.array(self.segments[subject]['preictal']), \
                    np.array(self.segments[subject]['pseudointerictal']), \
                    np.array(self.segments[subject]['pseudopreictal'])   ])
        else:
            self.training_segments = np.hstack([ \
                    np.array(self.segments[subject]['interictal']), \
                    np.array(self.segments[subject]['preictal'])])

        return X, y

    def _build_custom_X(self, subject, ictyp, featurearray):
        """
        Builds a custom x matrix for a subject and ictal type
        corresponding to the structure defined in the featurearray.
        Input:
        * subject
        * ictyp
        Output:
        * X
        """
        # iterate over features, calling _assemble feature
        # to build parts of the full X matrix
        X_parts = []
        feature_names = []
        for feature, index in featurearray:
            X_part = self._assemble_custom_feature(subject, feature, index, ictyp)
            X_parts += [X_part]

        # put these together with numpy
        X = np.hstack(X_parts)

        return X

    def _assemble_custom_feature(self, subject, feature, index, ictyp):
        """
        Create a matrix containing a feature vector in the order:
        Input:
        * subject
        * feature - which feature to build the matrix of
        * index - which index of this feature to use
        * ictyp
        Output:
        * X_part - part of the X matrix
        """

        # iterate over segments and build the X_part matrix
        rows = []
        print("Processing {0} with index {1}".format(feature, index))
        for segment in self.segments[subject][ictyp]:
            row = np.ndarray.flatten(self.data[feature][subject]\
                    [ictyp][segment])[int(index)]
            # gather up all the rows in the right order
            rows += [row]
        # stack up all the rows
        X_part = np.vstack(rows)
        pdb.set_trace()

        return X_part


    def build_custom_test(self, subject, namearray):
        """
        Takes a subject and array of names and feature indexes
        then builds a test set of precisely those feature elements,
        in that order.
        Input:
        * subject - string
        * namearray - array of structure [[feature_name, index],...]
        Output
        * X
        """

        return X


    def _composite_assemble_X(self, X_parts, dimensions):
        """
        Takes the parts of X and assembles into a large tiled X matrix:
         [X_part    X_part       X_part]
         becomes:
        [[X_part    nan          nan],
         [nan       X_part       nan],
         [nan       nan          X_part]]
        Input:
        * X_parts
        * dimensions (list of tuples of dimensions of the X_parts)
        Output:
        * X
        """
        # assemble this montrosity
        X = np.ones(np.sum(list(zip(*dimensions)), axis=1))*np.nan
        # assign each array within the new array, according to its size
        offset = [0, 0]
        for X_part in X_parts:
            d = X_part.shape
            X[offset[0]:offset[0]+d[0], offset[1]:offset[1]+d[1]] = X_part
            offset[0] += d[0]
            offset[1] += d[1]

        return X

    def composite_tiled_training(self):
        """
        Builds a composite tiled training set:
        Output:
        * X,y - tiled dataset:
        """
        # first assemble the pieces to build the tiled set from
        X_parts = []
        y_parts = []
        dimensions = []
        segments = []
        for subject in self.settings['SUBJECTS']:
            X, y = self.build_training(subject)
            X_parts += [X]
            y_parts += [y]
            dimensions += [X.shape]
            segments += [self.training_segments[:]]

        X = self._composite_assemble_X(X_parts, dimensions)

        # stack up y
        y = np.hstack(y_parts)

        # record of segments
        self.composite_training_segments = np.hstack(segments)

        # pending record of feature indexes

        return X, y

    def composite_tiled_test(self):
        """
        Builds a composite tiled training set:
        Output:
        * X,y - tiled dataset:
        """
        # first assemble the pieces to build the tiled set from
        X_parts = []
        dimensions = []
        segments = []
        for subject in self.settings['SUBJECTS']:
            X = self.build_test(subject)
            X_parts += [X]
            dimensions += [X.shape]
            segments += [self.test_segments[:]]

        X = self._composite_assemble_X(X_parts, dimensions)

        # keep record of feature indexes
        self.composite_test_segments = np.hstack(segments)

        return X

    def hour_classification_training(self, subject):
        """
        Builds a training set for testing classifiers
        at the task of predicting which hour a segment
        is from.
        Not compatible with our own SequenceCV.
        Input:
        * subject - string
        Output:
        * X,y
        """
        # modularity at work
        X, y = self.build_training(subject)

        hourIDs = []
        # then redefine the y vector based on the segments
        for segment in self.training_segments:
            segment = segment.split('.')[0]
            hourIDs.append(self.metadata[segment]['hourID'])
        y = np.array(hourIDs)
        return X, y

    def test_train_discrimination(self, subject):
        """
        Builds a training set for testing classifiers
        at the task of predicting whether a segment
        came from training or test.
        Not compatible with our own SequenceCV,
        but should be valid to apply any shuffled CV
        split.
        Input:
        * subject - string
        Output:
        * X,y
        """
        # modularity at work
        self.Xtrain, _ = self.build_training(subject)
        self.Xtest = self.build_test(subject)

        # stack the X matrices
        X = np.vstack([self.Xtrain, self.Xtest])

        # then redefine the y vector based on  test/train
        y = np.array([0]*self.Xtrain.shape[0] + [1]*self.Xtest.shape[0])

        return X, y


class Sequence_CV:
    def __init__(self, segments, metadata, r_seed=None, n_iter=10):
        """Takes a list of the segments ordered as they are in the array.
        Yield train,test tuples in the style of a sklearn iterator.
        Leave 50% out"""
        # put together the iterator
        # first make a dictionary mapping from the segments to which hour each is within
        self.segments = list(map(lambda segment: segment.split(".")[0], segments))

        self.seg2hour = {}
        self.seg2class = {}
        self.hour2class = {}
        # Loop over all segments in
        for segment in self.segments:
            # Look up the hourID number for this segment from the metadata
            hourID = int(metadata[segment]['hourID'])
            ictyp = metadata[segment]['ictyp']
            # dictionary identifying which class each segment should be in:
            if ictyp == 'preictal' or ictyp == 'pseudopreictal':
                # Record the class of this segment
                self.seg2class[segment] = 1
                # Record the hourIDstr of this segment, noting it is preictal
                self.seg2hour[segment] = "p{0}".format(hourID)
            elif ictyp == 'interictal' or ictyp == 'pseudointerictal':
                # Record the class of this segment
                self.seg2class[segment] = 0
                # Record the hourIDstr of this segment, noting it is interictal
                self.seg2hour[segment] = "i{0}".format(hourID)
            else:
                warnings.warn("Unfamiliar ictal type {0} in training data.".format(ictyp))
                continue
            # Make sure the hourIDstr of which this segment is a member is
            # in the mapping from hourIDstr to class
            self.hour2class[self.seg2hour[segment]] = self.seg2class[segment]

        # Find what unique hourIDstrings there are (p1, p2, ..., i1, i2, ...)
        self.hourIDs = np.array(list(set(self.seg2hour.values())))

        # array was unordered due to dictionary values method therefore CV
        # was returning inconsistent results on same data
        self.hourIDs.sort()

        # Presumably we need this line to make sure ordering is the same?
        y = [self.hour2class[hourID] for hourID in self.hourIDs]

        # ensure a good split by putting exactly one hour in test
        # Divide the no. of preictal hours by the total hours. Returns
        # number of samples to have exactly one preictal hour in test
        test_size = int(len(y)/sum(y))

        # Initialise a Stratified shuffle split
        self.cv = sklearn.cross_validation.StratifiedShuffleSplit(y,
                                                          n_iter=n_iter,
                                                          test_size=test_size,
                                                          random_state=r_seed)

        # Some of the datasets only have 3 hours of preictal recordings.
        # This will provie 10 stratified shuffles, each using 1 of the
        # preictal hours
        # Doesn't guarantee actually using each hour at least once though!
        # We fix the random number generator so we will always
        # use the same set of splits for this subject across
        # multiple CV tests for a fairer comparison.
        return None

    def __iter__(self):
        for train, test in self.cv:
            # map these back to the indices of the hourID list
            trainhourIDs = self.hourIDs[train]
            testhourIDs = self.hourIDs[test]
            train, test = [], []
            # Loop over all segments
            for i, segment in enumerate(self.segments):
                # Check if the hourID string is in the train or test partition
                hourID = self.seg2hour[segment]
                if hourID in trainhourIDs:
                    # I do not know generators, but this looks REALLY WRONG to me
                    # Surely this is adding each segment index to the list of
                    # hour indices which is the value of train as provided by self.cv?!
                    train.append(i)
                elif hourID in testhourIDs:
                    test.append(i)
                else:
                    warnings.warn("Unable to match {0} to train or test".format(segment))
            yield train, test

    def __len__(self):
        """
        Hidden function, should return number of CV folds.
        """
        return self.cv.__len__()


def subjsort_prediction(prediction_dict):
    '''
    Take the predictions and organise them so they are normalised for the number
    of preictal and interictal segments in the test data
    '''

    # Loop over all segments
    # for segment in prediction_dict.keys():
        # Look at segment and take out the subject name
        # Use this to split predictions by subject name
    # Within each subject, sort the segments by prediction value



    # Using prior knowledge about how many preictal and interictal segments we
    # expect to see, intersperse segments from each subject.
    # Allow prediction values to control local order, but maintain the
    # appropriate interspersion at the larger scale.

    # Replace prediciton values with (index within the sort)/(numsegments-1)
    return None


def output_csv(prediction_dict, settings, verbose=False):
    '''
    Parse the predictions and output them in the correct format
    for submission to the output directory
    input:  prediction_dict (dictionary of all predictions of the test data)
            settings (the settings dict from parsing the json_object)
    output: void
    '''
    output_file_basename = "{0}_submission_using_{1}_feats.csv".format(\
                                                settings['RUN_NAME'],
                                                settings['VERSION'])

    output_file_path = os.path.join(settings['SUBMISSION_PATH'],
                                    output_file_basename)

    print_verbose("@@Writing test probabilities to {0}".format(output_file_path),
                  flag=verbose)

    with open(output_file_path, 'w') as output_fh:
        csv_output = csv.writer(output_fh)
        csv_output.writerow(['clip', 'preictal'])
        for segment in prediction_dict.keys():
            # write segment idea and second probability as this
            # corresponds to the prob of class 1 (preictal)
            csv_output.writerow([segment,
                               str(prediction_dict[segment][-1])])


def get_cross_validation_set(y, *params):
    '''
    Return the cross_validation dataset
    input: y (target vector as np.array)
          *params
    output: cv (cross validation set)
    '''
    if params is None:
        cv = sklearn.cross_validation.StratifiedSuffleSplit(y)
        return cv

    cv = sklearn.cross_validation.StratifiedShuffleSplit(y, *params)

    return cv


def get_selector(settings):
    '''
    Return a sklearn selector object
    will __always__ use ANOVA f-values for selection
    input: **kwargs for selector params e.g. k
    output: sklearn.feature_selection object
    '''
    if 'KBEST' in settings['SELECTION'].keys():
        selector = sklearn.feature_selection.SelectKBest(\
                sklearn.feature_selection.f_classif,
                settings['SELECTION'])
    elif 'PERCENTILE' in settings['SELECTION'].keys():
        selector = sklearn.feature_selection.SelectPercentile(\
                sklearn.feature_selection.f_classif,
                percentile=settings['SELECTION']['PERCENTILE'])
    elif 'FOREST' in settings['SELECTION'].keys():
        selector = sklearn.ensemble.ExtraTreesClassifier(\
                n_estimators=settings['SELECTION']['FOREST'])
    elif 'SVC' in settings['SELECTION'].keys():
        # default from sklearn docs
        selector = sklearn.svm.LinearSVC(C=0.1, penalty="l1", dual=False)
    else:
        raise ValueError("Invalid feature selection"
                " option: {0}".format(settings['SELECTION']))

    return selector


def build_model_pipe(settings):
    '''
    Function to build and return the classification pipeline
    based on the values passed in the settings json
    '''

    pipe_elements = []

    # always use the standard scaler with default params
    scaler = sklearn.preprocessing.StandardScaler()
    pipe_elements.append(('scl', scaler))

    if 'THRESHOLD' in settings.keys():
        thresh = sklearn.feature_selection.VarianceThreshold()
        pipe_elements.append(('thr', thresh))

    if 'PCA' in settings.keys():
        pca_decomp = sklearn.decomposition.PCA(**settings['PCA'])
        pipe_elements.append(('pca', pca_decomp))

    if 'SELECTION' in settings.keys():
        selector = get_selector(settings)
        pipe_elements.append(('sel', selector))

    if 'TREE_EMBEDDING' in settings.keys():
        tree_embedding = sklearn.ensemble.RandomTreesEmbedding( \
                **settings['TREE_EMBEDDING'])
        pipe_elements.append(('embed', tree_embedding))

    classifier = settings['CLASSIFIER']

    if 'BAGGING' in settings.keys():
        # put the classifier in the bag, and append the bag
        bagging = sklearn.ensemble.BaggingClassifier(base_estimator=classifier,
                **settings['BAGGING'])
        pipe_elements.append(('clf', bagging))
    else:
        pipe_elements.append(('clf', classifier))

    model = sklearn.pipeline.Pipeline(pipe_elements)

    return model


def get_weights(y, settings={}):
    '''
    Take the y (target) vector and produce weights:
    input: y (target vector)
    output: weights vector
    '''
    if 'CUSTOM_WEIGHTING' in settings:
        weight = settings['CUSTOM_WEIGHTING']
    else:
        # calculate correct weighting for unbalanced classes
        weight = len(y)/sum(y)
    # generate vector for this weighting
    weights = np.array([weight if i == 1 else 1 for i in y])
    return weights


def get_metadata():
    '''
    Return the metadata.
    '''
    with open('segmentMetadata.json') as metafile:
        metadata = json.load(metafile)
    return metadata


def output_auc_scores(auc_scores, settings):
    '''
    Outputs AUC scores to a csv file in AUC_SCORE_PATH
    input: auc_score dict mapping subj to scores
            settings
    output: void
    '''
    # hacking this to save discriminate results
    if 'DISCRIMINATE' in settings:
        auc_csv_path = os.path.join(settings['AUC_SCORE_PATH'],
                                    'discriminate_scores.csv')
    else:
        auc_csv_path = os.path.join(settings['AUC_SCORE_PATH'],
                                    'AUC_scores.csv')

    colnames = [subj for subj in settings['SUBJECTS']] + ['all']

    scores = [auc_scores[subj] for subj in colnames]

    auc_row = [settings['RUN_NAME']] + scores

    # Add more details to the CSV
    colnames.append('has_pseudo')
    if any("pseudo" in s for s in settings['DATA_TYPES']):
        auc_row.append('y')
    else:
        auc_row.append('n')

    colnames.append('classifier')
    auc_row.append(re.sub('\s+', ' ', settings['CLASSIFIER'].__str__()))

    colnames.append('features')
    auc_row.append(';'.join(settings['FEATURES']))

    # if csv exists just append the new auc line for this run

    if os.path.exists(auc_csv_path):
        with open(auc_csv_path, 'a') as auc_csv:
            writer = csv.writer(auc_csv, delimiter="\t")
            writer.writerow(auc_row)
    else:
        with open(auc_csv_path, 'a') as auc_csv:
            writer = csv.writer(auc_csv, delimiter="\t")
            writer.writerow(['RUN_NAME'] + colnames)
            writer.writerow(auc_row)


def mvnormalKL(mu_0, mu_1, Sigma_0, Sigma_1):
    """
    Takes the parameters of two multivariate normal distributions
    and calculates the KL divergence between them.
    Equation taken straight from [Wikipedia][].
    Input:
    * mu_0 - mean vector of first distribution
    * mu_1 - mean vector of second distribution
    Output:
    * Sigma_0 - covariance matrix of first distribution
    * Sigma_1 - covariance matrix of second distribution

    Currently __broken__ when used in our code, giving
    negative KL divergences.

    [wikipedia]: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
    """
    if len(mu_0.shape) < 2 or len(mu_1.shape) < 2:
        raise ValueError("Mean vectors must be column vectors.")
    K = mu_0.shape[0]
    if mu_1.shape[0] != K:
        raise ValueError("Mean vectors must share the same dimension.")
    KL = float(0.5*(np.trace( np.dot( np.linalg.inv(Sigma_1), Sigma_0 )  ) \
            + np.dot((mu_1 - mu_0).T, np.dot(np.linalg.inv(Sigma_1), (mu_1 - mu_0)))  \
            - K - (np.prod(np.linalg.slogdet(Sigma_0)) - \
            np.prod(np.linalg.slogdet(Sigma_1)))))
    if KL.__repr__() == 'nan':
        raise ValueError
    else:
        return KL


def reliability_plot(predictions, labels):
    """
    Returns lists for plotting a reliability chart.
    Does not actually plot it for you.
    Splits into bins and plots the mean predicted value
    against the true fraction of positive cases.
    Input:
    * predictions
    * labels
    Output:
    * x
    * y
    """
    # split both arrays into bins:
    edges = np.linspace(0, 1, 21)
    # store results in nice simple lists
    x = []
    y = []
    # this is very inefficient, but it doesn't matter
    # improves readability
    for ledge, hedge in zip(edges[:-1], edges[1:]):
        binned_predictions = []
        binned_labels = []
        for prediction, label in zip(predictions, labels):
            if prediction > ledge and prediction < hedge:
                binned_predictions.append(prediction)
                binned_labels.append(label)
        # avoid empty bins
        if binned_predictions != []:
            mean_predicted = np.mean(binned_predictions)
            fraction_positive = sum(binned_labels)/len(binned_labels)
            x.append(mean_predicted)
            y.append(fraction_positive)

    return x, y


def get_feature_ids(names, pickled=None):
    """
    Takes an array of feature names and
    processes it to create an array of
    tuples in which each tuple also contains
    the index of that feature.
    Useful for later being able to find this
    exact feature element.
    Input:
    * names - array of feature names
    Output:
    * feature_ids - 2d array of feature names and indices
    """
    # iterate over array, incrementing counter whenever
    # feature does not change
    indices = []
    counter = 0
    prevname = names[0]
    for name in names:
        if name == prevname:
            indices.append(counter)
            counter += 1
        else:
            prevname = name
            counter = 0
            indices.append(counter)
    indices = np.array(indices)[np.newaxis].T
    names = names[np.newaxis].T
    feature_ids = np.hstack([names, indices])
    if pickled != None:
        # add extra pickled ids
        feature_ids = np.vstack([feature_ids, pickled])
    return feature_ids


def train_RFE(settings, data, metadata, subject, model_pipe,
              transformed_features, store_models, store_features,
              load_pickled, settingsfname, verbose, extra_data=None):

    # initialise the data assembler
    assembler = DataAssembler(settings, data, metadata)
    X, y = assembler.build_training(subject)

    if load_pickled:
        if extra_data is None:
            raise ValueError
        picklednames =  extra_data[subject]['names']
        X = np.hstack([X, extra_data[subject]['features']])

    # get the CV iterator
    cv = Sequence_CV(assembler.training_segments,
                           metadata,
                           r_seed=settings['R_SEED'],
                           n_iter=settings['CVITERCOUNT'])

    # initialise lists for cross-val results
    predictions = []
    labels = []
    allweights = []
    segments = []

    # first have to transform
    Xt = model_pipe.named_steps['scl'].fit_transform(X)
    if 'thr' in [step[0] for step in model_pipe.steps]:
        Xt = model_pipe.named_steps['thr'].fit_transform(Xt)
    # we might have huge numbers of features, best to remove in large numbers
    stepsize = int(Xt.shape[1]/20)
    rfecv = sklearn.feature_selection.RFECV(estimator=model_pipe.named_steps['clf'],
        step=stepsize, cv=cv, **settings['RFE'])
    rfecv.fit(Xt, y)
    # take the best grid score as the auc
    auc = max(rfecv.grid_scores_)

    if store_models:
        weights = get_weights(y)

        elements = []
        elements.append(('scl', model_pipe.named_steps['scl']))
        if 'thr' in [step[0] for step in model_pipe.steps]:
            elements.append(('thr', model_pipe.named_steps['thr']))
        elements.append(('clf', rfecv))
        model = sklearn.pipeline.Pipeline(elements)
        serialise_trained_model(model,
                                subject,
                                settings,
                                verbose=verbose)
    if store_features:
    # store a transformed version of the features
    # while at the same time keeping a log of where they came from
        mask = rfecv.support_
        # Storing as a dictionary using subjects as keys.
        # Inside each dictionary will be a dictionary
        # storing the transformed array and an index
        # describing which feature is which.
        feature_ids = get_feature_ids(assembler.training_names,\
                pickled=picklednames)
        feature_ids = feature_ids[mask]
        Xt = rfecv.transform(Xt)
        transformed_features[subject] = {'features':Xt,
                'names':feature_ids}
        # then pickle it
        if isinstance(store_features, str):
            with open(store_features+".pickle", "wb") as fh:
                pickle.dump(transformed_features, fh)
        else:
            with open(settingsfname.split(".")[0]
                    + "_feature_dump.pickle", "wb") as fh:
                pickle.dump(transformed_features, fh)

    return transformed_features, auc


def train_model(settings, data, metadata, subject, model_pipe,
                store_models, load_pickled, verbose, extra_data=None,
                parallel=None):
    # initialise the data assembler
    assembler = DataAssembler(settings, data, metadata)
    X, y = assembler.build_training(subject)

    if load_pickled:
        if extra_data is None:
            raise ValueError
        X = np.hstack([X, extra_data[subject]['features']])


    # get the CV iterator
    cv = Sequence_CV(assembler.training_segments,
                           metadata,
                           r_seed=settings['R_SEED'],
                           n_iter=settings['CVITERCOUNT'])

    # initialise lists for cross-val results
    predictions = []
    labels = []
    allweights = []
    segments = []

    # run cross validation and report results
    for train, test in cv:

        # calculate the weights
        weights = get_weights(y[train], settings=settings)
        # fit the model to the training data
        model_pipe.fit(X[train], y[train], clf__sample_weight=weights)
        # append new predictions
        predictions.append(model_pipe.predict_proba(X[test]))
        # append test weights to store (why?) (used to calculate auc below)
        weights = get_weights(y[test], settings=settings)
        allweights.append(weights)
        # store true labels
        labels.append(y[test])
        # store segments
        segments.append(assembler.training_segments[test])

    # stack up the results
    predictions = np.vstack(predictions)[:, 1]
    labels = np.hstack(labels)
    weights = np.hstack(allweights)
    segments = np.hstack(segments)

    # calculate the total AUC score
    auc = sklearn.metrics.roc_auc_score(labels,
                                        predictions,
                                        sample_weight=weights)

    print("predicted AUC score for {1}: {0:.2f}".format(auc, subject))

    if store_models:

        store_weights = get_weights(y, settings=settings)
        model_pipe.fit(X, y, clf__sample_weight=store_weights)
        serialise_trained_model(model_pipe,
                                      subject,
                                      settings,
                                      verbose=verbose)

    # store results from each subject

    results = (predictions, labels, weights, segments)

    if parallel:
        results = {subject: results}
        auc = {subject: auc}
        return (results, auc)

    return results, auc


def train_custom_model(settings, data, metadata, subject, model_pipe,
                store_models, load_pickled, verbose, extra_data=None):
    # initialise the data assembler
    assembler = DataAssembler(settings, data, metadata)
    # load the pickled array
    with open(settings['CUSTOM'], "rb") as fh:
        rfe_feature_dict = pickle.load(fh)
    featurearray = rfe_feature_dict[subject]['names']
    X, y = assembler.build_custom_training(subject, featurearray)

    if load_pickled:
        if extra_data is None:
            raise ValueError
        X = np.hstack([X, extra_data[subject]['features']])


    # get the CV iterator
    cv = Sequence_CV(assembler.training_segments,
                           metadata,
                           r_seed=settings['R_SEED'],
                           n_iter=settings['CVITERCOUNT'])

    # initialise lists for cross-val results
    predictions = []
    labels = []
    allweights = []
    segments = []

    # run cross validation and report results
    for train, test in cv:

        # calculate the weights
        weights = get_weights(y[train])
        # fit the model to the training data
        model_pipe.fit(X[train], y[train], clf__sample_weight=weights)
        # append new predictions
        predictions.append(model_pipe.predict_proba(X[test]))
        # append test weights to store (why?) (used to calculate auc below)
        weights = get_weights(y[test])
        allweights.append(weights)
        # store true labels
        labels.append(y[test])
        # store segments
        segments.append(assembler.training_segments[test])

    # stack up the results
    predictions = np.vstack(predictions)[:, 1]
    labels = np.hstack(labels)
    weights = np.hstack(allweights)
    segments = np.hstack(segments)

    # calculate the total AUC score
    auc = sklearn.metrics.roc_auc_score(labels,
                                        predictions,
                                        sample_weight=weights)

    print("predicted AUC score for {1}: {0:.2f}".format(auc, subject))

    if store_models:

        store_weights = get_weights(y)
        model_pipe.fit(X, y, clf__sample_weight=store_weights)
        serialise_trained_model(model_pipe,
                                      subject,
                                      settings,
                                      verbose=verbose)

    # store results from each subject

    results = (predictions, labels, weights, segments)
    return results, auc


def combined_auc_score(settings, auc_scores, subj_pred=None):

    if 'RFE' in settings:
        combined_auc = np.mean(list(auc_scores.values()))
    else:
        if subj_pred is None:
            raise ValueError('Subject prediction dict needs to not be None')

        # stack subject results (don't worry about this line)
        predictions, labels, weights, segments = map(np.hstack,
                                         zip(*list(subj_pred.values())))

        # calculate the total AUC score over all subjects
        combined_auc = sklearn.metrics.roc_auc_score(labels, predictions)
                                                    # sample_weight=weights)

    return combined_auc


def round_sf(x, n=1):
    '''
    Round to the nearest significant figure.
    '''
    return round(x, -int(math.floor(math.log10(abs(x))))+n-1)
