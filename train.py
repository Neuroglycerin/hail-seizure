#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json
from sklearn.pipeline import Pipeline

def get_data(subjects):
    return [utils.parse_matlab_HDF5(subj_dataset) for subj_dataset in subjects]

if __name__=='__main__':

    settings = json.load(open('SETTINGS.json', 'r'))

    data = get_data(settings['SUBJECTS'])
