import json
import numpy as np
import os
import h5py
from sklearn.externals import joblib #pickle w/ optimisation for np arrays
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.pipeline
import sklearn.ensemble

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

    # Try to open hdf5 file if it doesn't exist print error and return None
    try:
        h5_from_matlab = h5py.File(h5_file_name, 'r')
    except IOError:
        print("WARNING: {0} does not exist (or is not readable)".format(h5_file_name))
        return None

    # parse h5 object into dict (see docstring for struct)

    feature_dict = {}
    try:
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
    except:
        print("WARNING: Unable to parse {0}".format(h5_file_name))


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
            data (data structure produced by get_data)
    output: X (feature matrix as np.array)
            y (target vector as np.array)
            cross-validation iterator

    Does not preserve feature names
    '''
    # start by loading the metadata about the sequences
    with open('segmentMetadata.json') as metafile:
        metadata = json.load(metafile)

    segments = 'empty'
    # hacking this for later
    first = features[0]
    for feature in features:
        Xf = np.array([])

        # enumerate to get numbers for target vector:
        #     0 is interictal
        #     1 is preictal
        for i,ictal in enumerate(['interictal','preictal']):
            # this is bona fide programming
            if segments == 'empty':
                segments = [np.array(list(data[feature][subject][ictal].keys())),[]]
            elif segments[1] == []:
                segments[1] = np.array(list(data[feature][subject][ictal].keys()))
            for segment in segments[i]:
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

    # stick together the segments
    segments = np.hstack(segments)
    
    # create CV iterator
    cv = Sequence_LOO_CV(segments,metadata)

    # turn y into an array
    y = np.array(y)
    return X,y,cv,segments

class Sequence_LOO_CV:
    def __init__(self,segments,metadata):
        """Takes a list of the segments ordered as they are in the array.
        Yield train,test tuples in the style of a sklearn iterator"""
        # put together the iterator
        # first make a dictionary mapping from the segments to which sequence each is found in
        self.segments = list(map(lambda segment: segment.split(".")[0], segments))
        self.segtoseq = {}
        for segment in self.segments:
            # doubling and converting to int here because comparing floats is problematic
            # have to double due to x.5 sequence numbers for pseudo-data
            self.segtoseq[segment] = int(2*metadata[segment]['seqence'])
        # find what sequences we're working with
        self.sequences = np.array(list(set(self.segtoseq.values())))
        self.cv = sklearn.cross_validation.LeaveOneOut(len(self.sequences))
        return None

    def __iter__(self):
        for train,test in self.cv:
            # map these back to the indices of the segments in each sequence
            trainsequences = self.sequences[train]
            testsequences = self.sequences[test]
            train,test = [],[]
            for i,segment in enumerate(self.segments):
                sequence = self.segtoseq[segment]
                if sequence in trainsequences:
                    train.append(i)
                elif sequence in testsequences:
                    test.append(i)
                else:
                    print("Warning, unable to match {0} to train or test.".format(segment))
            yield train,test       

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
            fvector = np.ndarray.flatten(data[feature][subject]['test'][segment])
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
    selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif,
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

def get_classifier(**kwargs):
    '''
    Return the classifier object
    input: **kwargs for classifier params
    output: sklearn.ensemble classifier object
    '''
    classifier = sklearn.ensemble.RandomForestClassifier(**kwargs)
    return classifier


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
