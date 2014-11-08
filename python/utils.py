import json
import numpy as np
import os
import h5py
import csv
import warnings
from sklearn.externals import joblib #pickle w/ optimisation for np arrays
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.pipeline
import sklearn.ensemble
import sklearn.svm
import optparse

def get_train_parser():
    '''
    Generate optparse parser object for train.py
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

    parser.add_option("-s", "--settings",
                      action="store",
                      dest="settings",
                      default="SETTINGS.json",
                      help="Settings file to use in JSON format (default="
                            "SETTINGS.json)")

    parser.add_option("-k", "--selector_k",
                      action="store",
                      dest="selector_k",
                      type=int,
                      default="3000",
                      help="Number of best features to select"\
                           " via ANOVA f-scores (default=3000)")

    parser.add_option("-d", "--max_depth",
                      action="store",
                      dest="max_depth",
                      type=int,
                      default="3",
                      help="Max tree depth in random forest classifier"
                           "(default=3)")

    parser.add_option("-t", "--trees",
                      action="store",
                      dest="tree_num",
                      type=int,
                      default="100",
                      help="Number of estimators to use in random forest classifier"
                           "(default=100)")

    parser.add_option("-j", "--cores",
                      action="store",
                      dest="cores",
                      type=int,
                      default=-1,
                      help="Number of cores to use when training classifier"
                           " (default is all of them)")

    return parser

def get_predict_parser():
    '''
    Generate optparse parser object for predict.py
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

    parser.add_option("-s", "--settings",
                      action="store",
                      dest="settings",
                      default="SETTINGS.json",
                      help="Settings file to use in JSON format (default="
                            "SETTINGS.json)")
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


    # update file paths settings to have full absolute paths
    for settings_field in ['TRAIN_DATA_PATH',
                           'MODEL_PATH',
                           'TEST_DATA_PATH',
                           'SUBMISSION_PATH']:

        settings[settings_field] = os.path.abspath(settings[settings_field])

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
    if type(flag) is not bool:
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
                         feature_dict[subj][typ]=h5_from_matlab[subj][typ].value

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
    joblib.dump(model,
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
    model = joblib.load(os.path.join(settings['MODEL_PATH'], model_name))

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

        # parse for segment tuple/list
        self.segments = self._parse_segment_names()

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
        # This will fix the order of the segments
        # iterate over all possible _training_ segments
        for segment in self.metadata.keys():
            # for this segment, find what subject it's in
            subject = self.metadata[segment]['subject']
            # and what ictyp it is
            ictyp = self.metadata[segment]['ictyp']
            # append .mat to match strings in data
            segment = segment + '.mat'
            # store in the dictionary of dictionaries
            if ictyp in self.settings['DATA_TYPES']:
                segments[subject][ictyp] += [segment]

        # this will iterate over all possible test segments:
        feature = self.settings['FEATURES'][0]
        ictyp = 'test'
        for subject in self.settings['SUBJECTS']:
            for segment in self.data[feature][subject][ictyp].keys():
                segments[subject][ictyp] += [segment]

        # then enforce tuple
        for subject in segments.keys():
            for ictyp in segments[subject].keys():
                segments[subject][ictyp] = tuple(segments[subject][ictyp])

        return segments

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
            X_part = self._assemble_feature(subject,feature,ictyp)
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
        # iterate over segments and build the X_part matrix
        rows = []
        for segment in self.segments[subject][ictyp]:
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
        X_inter,self.training_names = self._build_X(subject,'interictal')
        X_pre,verification_names = self._build_X(subject,'preictal')

        if all(tr != vf for tr in self.training_names for \
                vf in verification_names):
            raise ValueError
        X = np.vstack([X_inter,X_pre])
        y = np.hstack([self._build_y(subject,'interictal'), \
                self._build_y(subject,'preictal')])
        # storing feature names in self.training_names

        # storing the correct sequence of segments
        self.training_segments = np.hstack([ \
                np.array(self.segments[subject]['interictal']), \
                np.array(self.segments[subject]['preictal'])])

        return X,y


    def build_test(self, subject):
        """
        Builds test set for given subject.
        Input:
        * subject
        Output:
        * X
        """
        # storing names for the features in self.test_names
        X,self.test_names = self._build_X(subject,'test')

        # storing the correct sequence of segments
        self.test_segments = np.array(self.segments[subject]['test'])

        return X

    def _composite_assemble_X(self,X_parts,dimensions):
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
        offset = [0,0]
        for X_part in X_parts:
            d = X_part.shape
            X[offset[0]:offset[0]+d[0],offset[1]:offset[1]+d[1]] = X_part
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
        for subject in self.settings['SUBJECTS']:
            X,y = self.build_training(subject)
            X_parts += [X]
            y_parts += [y]
            dimensions += [X.shape]

        X = self._composite_assemble_X(X_parts,dimensions) 

        # stack up y
        y = np.hstack(y_parts)

        # pending record of feature indexes

        return X,y

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

        X = self._composite_assemble_X(X_parts,dimensions) 
        
        # keep record of feature indexes
        self.composite_test_segments = np.hstack(segments)

        return X


class Sequence_CV:
    def __init__(self, segments, metadata, r_seed=None):
        """Takes a list of the segments ordered as they are in the array.
        Yield train,test tuples in the style of a sklearn iterator.
        Leave 20% out"""
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

        # need to build a y vector for these sequences

        # Presumably we need this line to make sure ordering is the same?
        y = [self.hour2class[hourID] for hourID in self.hourIDs]

        # Initialise a Stratified shuffle split
        self.cv = sklearn.cross_validation.StratifiedShuffleSplit(y,
                                                                  n_iter=10,
                                                                  test_size=0.2,
                                                                  random_state=r_seed)

        # Some of the datasets only have 3 hours of preictal recordings.
        # This will provie 10 stratified shuffles, each using 1 of the preictal hours
        # Doesn't guarantee actually using each hour at least once though!
        # We fix the random number generator so we will always use the same split
        # for this subject across multiple CV tests for a fairer comparison.
        return None

    def __iter__(self):
        for train,test in self.cv:
            # map these back to the indices of the hourID list
            trainhourIDs = self.hourIDs[train]
            testhourIDs = self.hourIDs[test]
            train,test = [],[]
            # Loop over all segments
            for i,segment in enumerate(self.segments):
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


def build_test(subject, features, data):
    '''
    Function to build data structures for submission
    input : subject  (subject name string)
            features (features to use)
            data (nested dicts datastructure of all feature extracted data)
    output: X (feature matrix as np.array)
            y (labels as np.array)
    '''

    segments =  'empty'
    Xd = {}
    for feature in features:
        if segments == 'empty':
            segments = data[feature][subject]['test'].keys()
        for segment in segments:
            fvector = np.ndarray.flatten(\
                    data[feature][subject]['test'][segment])
            try:
                Xd[segment] = np.hstack([Xd[segment], fvector])
            except:
                Xd[segment] = fvector

    # make the X array and corresponding labels
    X = []
    for segment in segments:
        X.append(Xd[segment])
    X = np.vstack(X)
    return X, segments


def subjsort_prediction(prediction_dict):
    '''
    Take the predictions and organise them so they are normalised for the number
    of preictal and interictal segments in the test data
    '''

    # Loop over all segments
    #for segment in prediction_dict.keys():
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

def get_selector(**kwargs):
    '''
    Return a sklearn selector object
    will __always__ use ANOVA f-values for selection
    input: **kwargs for selector params e.g. k
    output: sklearn.feature_selection object
    '''
    selector = sklearn.feature_selection.SelectKBest(\
            sklearn.feature_selection.f_classif,
            **kwargs)

    return selector

def get_scaler(**kwargs):
    '''
    Return a sklearn scaler object
    input: **kwargs for scaler params
    output sklearn.preprocessing scaler object
    '''
    scaler = sklearn.preprocessing.StandardScaler(**kwargs)
    return scaler

def get_model(elements):
    '''
    Assemble the pipeline for classification

    input: elements in the following structure:
            [('scl',scaler), ..., ('clf',classifier)]
    output model - sklearn.pipeline object model
    '''
    model = sklearn.pipeline.Pipeline(elements)
    return model

def fit_model(model_pipe, X, y, cv, **kwargs):
    '''
    Fit provided model using pipeline and data
    input: model_pipe - sklearn.pipeline pipe
           X - feature vector
           y - target vector
           cv - cross validation set
           **kwargs - for the fitting
    output: fitted_model - model fitted to the dataset
    '''
    model.fit(X, y, **kwargs)

    return model

def get_thresh(**kwargs):
    '''
    Make a sklearn variance thresholder
    input: kwargs
    output: sklearn variance threshold object
    '''
    return sklearn.feature_selection.VarianceTheshold(**kwargs)

def get_weights(y):
    '''
    Take the y (target) vector and produce weights:
    input: y (target vector)
    output: weights vector
    '''
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
