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

    X,y = utils.build_training(settings['SUBJECTS'], features, data)

    cv = utils.get_cross_validation_set(y)

    selector = utils.get_selector(k=3000)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    model_pipe = utils.get_model(selector, scaler, classifier)

    for train, test in cv:
        fitted_model = utils.fit_model(model_pipe,
                                  X[train],
                                  y[train],
                                  cv,
                                  clf__sample_weight=weights)

        serialise_trained_model(fitted_model, "model_{0}".format(str(model_index)))
