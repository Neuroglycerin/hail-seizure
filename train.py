#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json
from sklearn.pipeline import Pipeline

def get_data(features):
    return {feat_name: utils.parse_matlab_HDF5(feat_name) for feat_name in features}

if __name__=='__main__':

    settings = json.load(open('SETTINGS.json', 'r'))
    features = ['raw_' + feat_name + '_' for feat_name in settings['FEATURES']]
    for feat in ['ica_' + feat_name + '_' for feat_name in settings['FEATURES']]:
        features.append(feat)
    data = get_data(features)
