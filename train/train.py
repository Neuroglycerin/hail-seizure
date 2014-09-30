import json
import numpy as np
import h5py


def parse_matlab_HDF5(subj):

    feature_location = json_settings['TRAIN_DATA_PATH']
    version = json_settings['VERSION']

    h5_from_matlab = h5py.File(feature_location + '/' + subj + version + '.h5',
                             'r')

    subject_dict = {typ: {feature: h5_from_matlab[typ][feature].value for feature in h5_from_matlab[typ]} for typ in h5_from_matlab}

    h5_from_matlab.close()

    return subject_dict


if __name__=='__main__':

    # Parse settings and make global
    with open('SETTINGS.json') as settings_fh:
        global json_settings
        json_settings = json.load(settings_fh)

