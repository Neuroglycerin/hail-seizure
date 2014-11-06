#!/usr/bin/env python3

import python.utils as utils #contains mainly parsers
import json
import glob
import train
import csv

def main(settings_file='SETTINGS.json'):

    #load the settings
    settings = utils.get_settings(settings_file)

    subjects = settings['SUBJECTS']
    features = settings['FEATURES']

    #load the data
    data = utils.get_data(features, settings)

    features_that_parsed = list(data.keys())
    #iterate over subjects
    prediction_dict = {}

    for subject in subjects:
        #load the trained model:
        model = utils.read_trained_model(subject, settings, verbose=opts.verbose)

        #build test set
        X, segments = utils.build_test(subject, features_that_parsed, data)
        #make predictions
        predictions = model.predict_proba(X)
        for segment, prediction in zip(segments, predictions):
            prediction_dict[segment] = prediction

    utils.output_csv(prediction_dict, settings, verbose=opts.verbose)

if __name__=='__main__':

    parser = utils.get_predict_parser()
    (opts, args) = parser.parse_args()

    main(settings_file=opts.settings)
