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
    data = utils.get_data(settings)

    #load the metadata
    metadata = utils.get_metadata()

    features_that_parsed = list(data.keys())
    settings['FEATURES'] = [feature for feature in settings['FEATURES'] \
            if feature in features_that_parsed]
    #iterate over subjects
    prediction_dict = {}

    for subject in subjects:
        #load the trained model:
        model = utils.read_trained_model(subject, settings, verbose=opts.verbose)

        #initialise the data assembler
        assembler = utils.DataAssembler(settings, data, metadata)
        #build test set
        X = assembler.build_test(subject)

        #make predictions
        predictions = model.predict_proba(X)
        for segment, prediction in zip(assembler.test_segments, predictions):
            prediction_dict[segment] = prediction

    utils.output_csv(prediction_dict, settings, verbose=opts.verbose)

if __name__=='__main__':

    parser = utils.get_predict_parser()
    (opts, args) = parser.parse_args()

    main(settings_file=opts.settings)
