#!/usr/bin/env python

import random
import unittest
import subprocess
import os
import python.utils as utils
import filecmp

class test_HDF5_parsing(unittest.TestCase):

    def setUp(self):
        self.settings_fh = 'test_settings.json'
        self.settings = utils.get_settings(self.settings_fh)

    def test_error(self):
        '''
        Assert a non-existent file correctly raises IOError
        '''
        self.assertRaises(IOError,
                          utils.parse_matlab_HDF5('fake_feat', self.settings))

    def test_parse(self):
        '''
        Very basic check of parsed object structure containing minimum 3 dicts
        '''
        parsed_HDF5 = utils.parse_matlab_HDF5(self.settings['FEATURES'][0],
                                              self.settings)

        self.assertIs(type(parsed_HDF5), dict)

        subj_layer = parsed_HDF5[list(parsed_HDF5.keys())[0]]
        self.assertIs(type(subj_layer), dict)

        typ_layer = subj_layer[list(subj_layer.keys())[0]]
        self.assertIs(type(typ_layer), dict)

class crude_model_test(unittest.TestCase):

    def setUp(self):

        self.settings_fh = 'test_settings.json'
        self.settings = utils.get_settings(self.settings_fh)

    def test_full_train_run(self):
        '''
        Crude test of train.py on dummy dataset
        '''
        # call full pipe as subprocess then compare dir to sample dir
        subprocess.call(['python', '../train.py', '-s', 'test_settings.json'])

        model_files = os.listdir('model')

        self.assertEqual(len(model_files), 7)

        output_model_stats = os.stat('model/model_Dog_1__v2')
        output_model_size = output_model_stats.st_size

        self.assertTrue(4000 < output_model_size < 5000)

    def tearDown(self):
        for f in os.listdir('model'):
            f_path = os.path.join('model', f)
            if os.path.isfile(f_path):
                os.unlink(f_path)

if __name__=='__main__':

    unittest.main()
