#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json

def get_data(features):
    return {feat_name: utils.parse_matlab_HDF5(feat_name) for feat_name in features}

if __name__=='__main__':

    settings = json.load(open('SETTINGS.json', 'r'))

    features = ['raw_' + feat_name + '_' for feat_name in settings['FEATURES']]
    for feat in ['ica_' + feat_name + '_' for feat_name in settings['FEATURES']]:
        features.append(feat)

    data = get_data(features)

    X,y = utils.build_training(list(settings['SUBJECTS'].keys()), features, data)

    # this cross-val is broken at the moment, for reasons discussed in the meeting
    cv = utils.get_cross_validation_set(y)

    thresh = utils.get_thresh()

    selector = utils.get_selector(k=3000)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    model_pipe = utils.get_model([('thr',thresh),('sel',selector),('scl',scaler),('cls',classifier)])

    model_index = 0

    for train, test in cv:
        model_index +=1
        fitted_model = utils.fit_model(model_pipe,
                                  X[train],
                                  y[train],
                                  cv,
                                  clf__sample_weight=weights)

        serialise_trained_model(fitted_model, "model_{0}".format(str(model_index)))
