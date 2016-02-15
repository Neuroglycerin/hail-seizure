#!/usr/bin/env python3

import python.utils as utils  # contains mainly parsers
import json
import glob
import train
import csv
import numpy as np


def main(settings_file='SETTINGS.json', verbose=False):

    # load the settings
    settings = utils.get_settings(settings_file)

    subjects = settings['SUBJECTS']
    features = settings['FEATURES']

    # load the data
    data = utils.get_data(settings)

    # load the metadata
    metadata = utils.get_metadata()

    features_that_parsed = list(data.keys())
    settings['FEATURES'] = [feature for feature in settings['FEATURES']
                            if feature in features_that_parsed]

    # check if features are 1-minute
    if all('10feat' in feature for feature in settings['FEATURES']):
        # set the flage
        minutefeatures = True
    elif not all('10feat' in feature for feature in settings['FEATURES']) and \
            any('10feat' in feature for feature in settings['FEATURES']):
        raise ValueError("Cannot mix 1-minute and 10-minute features.")
    else:
        minutefeatures = False

    # iterate over subjects
    prediction_dict = {}

    for subject in subjects:
        # load the trained model:
        model = utils.read_trained_model(
            subject,
            settings,
            verbose=verbose)

        # initialise the data assembler
        assembler = utils.DataAssembler(settings, data, metadata)
        # build test set
        X = assembler.build_test(subject)

        # make predictions
        predictions = model.predict_proba(X)

        # if using minute features combine the estimates
        # on each segment by averaging
        if minutefeatures:
            segmentdict = {}
            for segment, prediction in zip(assembler.test_segments, predictions):
                if segment not in segmentdict:
                    segmentdict[segment] = []
                segmentdict[segment].append(prediction)
            # gathered all predictions corresponding to a segment together
            # now average them along their columns
            for segment in assembler.test_segments:
                segmentdict[segment] = np.vstack(segmentdict[segment])
                segmentdict[segment] = np.mean(segmentdict[segment], axis=0)

        for segment, prediction in zip(assembler.test_segments, predictions):
            prediction_dict[segment] = prediction

    utils.output_csv(prediction_dict, settings, verbose=verbose)

if __name__ == '__main__':

    parser = utils.get_parser()
    args = parser.parse_args()

    main(settings_file=args.settings, verbose=args.verbose)
