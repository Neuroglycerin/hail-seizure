#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json

def get_data(features):
    return {feat_name: utils.parse_matlab_HDF5(feat_name) for feat_name in features}

if __name__=='__main__':

    settings = json.load(open('SETTINGS.json', 'r'))

    features = settings['FEATURES']

    data = get_data(features)

    subjects = list(settings['SUBJECTS'].keys())

    X,y = utils.build_training(subjects, features, data)

    # this cross-val is broken at the moment, for reasons discussed in the meeting
    cv = utils.get_cross_validation_set(y)

    thresh = utils.get_thresh()

    selector = utils.get_selector(k=3000)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    model_pipe = utils.get_model([('thr',thresh),('sel',selector),('scl',scaler),('cls',classifier)])

    for train, test,subject in zip(cv,subjects):
        fitted_model = utils.fit_model(model_pipe,
                                  X[train],
                                  y[train],
                                  cv,
                                  clf__sample_weight=weights)

        serialise_trained_model(fitted_model, "model_{0}_{1}".format(subject,setting["VERSION"]))

