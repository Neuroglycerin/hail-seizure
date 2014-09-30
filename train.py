import json
import numpy as np
import h5py


# Parse settings and make global
with open('SETTINGS.json') as settings_fh:
    global json_settings
    json_settings = json.load(settings_fh)

def parse_matlab_HDF5(subj):
    '''
    Parse h5 file from matlab into hierarchial dict containing np arrays
    input: subj - subject name (e.g. Dog1),
    output: subject_dict containing data in hierarchial format
                    e.g. Dog_1 = {'interictal':
                                        {'feat1': featvector (np.array),
                                         'feat2 : featvector (np.array),
                                         ...},
                                  'preictal':
                                        {'feat1': featvector (np.array),
                                         ...},
                                   'test': ...}
    '''


    feature_location = json_settings['TRAIN_DATA_PATH']
    version = json_settings['VERSION']

    # open h5 read-only file for correct subj and version number
    h5_file_name = "{0}/{1}{2}.h5".format(feature_location, subj, version)
    h5_from_matlab = h5py.File(h5_file_name, 'r')

    # parse h5 object into dict using nested comprehensions (see docstring
    # for struct)
    subject_dict = {typ: {feature: h5_from_matlab[typ][feature].value for feature in h5_from_matlab[typ]} for typ in h5_from_matlab}

    # make sure h5 object is closed
    h5_from_matlab.close()

    return subject_dict


