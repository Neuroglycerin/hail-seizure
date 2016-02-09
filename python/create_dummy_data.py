#!/usr/bin/env python3

import h5py
import sys
import numpy as np


def main(args):

    input_feat_path = args[1]
    new_feat_path = args[2]
    value = args[3]

    feat_template = h5py.File(input_feat_path, 'r')
    new_feat = h5py.File(new_feat_path, 'w')

    for subj in feat_template:
        for ictyp in feat_template[subj]:
            dataformat = type(feat_template[subj][ictyp])
            if dataformat is h5py._hl.group.Group:
                for seg in feat_template[subj][ictyp]:
                    dataset_dim = np.size(
                        feat_template[subj][ictyp][seg].value)
                    dataset = np.empty(dataset_dim)
                    dataset.fill(value)

                    new_feat.create_dataset(name='/'.join([subj, ictyp, seg]),
                                            data=dataset)

            elif dataformat is h5py._hl.dataset.Dataset:
                dataset_dim = np.size(feat_template[subj][ictyp].value)
                dataset = np.empty(dataset_dim)
                dataset.fill(value)
                new_feat.create_dataset(name='/'.join([subj, ictyp]),
                                        data=dataset)

    feat_template.close()
    new_feat.close()

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print(
            "Usage: create_dummy_data.py <PATH_TO_H5_YOU_WANT_TO_MIMIC> <NAME_OF_DUMMY_H5> <FILL VAL>")
        sys.exit()

    main(sys.argv)
