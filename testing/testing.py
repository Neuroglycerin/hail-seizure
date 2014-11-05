#!/usr/bin/env python

import random
import unittest
import subprocess
import os
import python.utils as utils
import csv

class test_HDF5_parsing(unittest.TestCase):

    def setUp(self):
        self.settings_fh = 'test_settings.json'
        self.settings = utils.get_settings(self.settings_fh)
        self.all_subjects = set(['Dog_1',
                             'Dog_2',
                             'Dog_3',
                             'Dog_4',
                             'Dog_5',
                             'Patient_1',
                             'Patient_2'])
        self.all_types = set(['preictal',
                              'MIerr',
                              'test',
                              'MI',
                              'pseudointerictal',
                              'interictal',
                              'pseudopreictal'])

    def test_ioerror(self):
        '''
        Assert a non-existent file correctly raises IOError
        '''
        self.assertRaises(IOError,
                          utils.parse_matlab_HDF5('fake_feat', self.settings))

    def test_hdf5_parse(self):
        '''
        Very basic check of parsed object structure containing minimum 3 dicts
        '''
        parsed_HDF5 = utils.parse_matlab_HDF5(self.settings['FEATURES'][0],
                                              self.settings)

        subjects = set(parsed_HDF5.keys())
        self.assertEqual(subjects, self.all_subjects)

        typs = set(parsed_HDF5['Dog_1'].keys())
        self.assertEqual(typs, self.all_types)

        num_interictal_dog1_segs = len(\
                parsed_HDF5['Dog_1']['interictal'].keys())

        self.assertEqual(num_interictal_dog1_segs, 480)


class test_train(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.settings_fh = 'test_settings.json'
        cls.settings = utils.get_settings(cls.settings_fh)

        f = open('stdout_tmp', 'w')
        cls.proc = subprocess.call(['../train.py',
                                      '-s', 'test_settings.json'],
                                      stdout=f)
        f.close()

        with open('stdout_tmp', 'r') as f:
            cls.stdout = f.read()

        cls.model_files = [x for x in \
                os.listdir(cls.settings['MODEL_PATH']) if x[0]!='x']

    def test_train_stdout(self):
        '''
        Test stdout prints correct number of AUC scores
        '''
        # count the number of AUC scores printed to stdout
        # and assert this is 8 (7 subjects and 1 overall)
        AUC_score_count = self.stdout.count('AUC')
        self.assertEqual(AUC_score_count, 8)

    def test_model_number(self):
        '''
        Test correct number of models are generated
        '''
        # get number of models
        self.assertEqual(len(self.model_files), 7)

    def test_model_size_correct(self):
        '''
        Test if one of the serialised models is roughly correct size
        '''
        # randomly pick an output model
        output_model = random.choice(self.model_files)

        # get file size and assert between 2.5 and 8k
        output_model_stats = os.stat(os.path.join(\
                self.settings['MODEL_PATH']+'/'+output_model))
        output_model_size = output_model_stats.st_size
        self.assertTrue(2500 < output_model_size < 8000)

    def test_model_can_be_read(self):
        '''
        Check whether a model can be read
        '''
        output_model = random.choice(self.model_files)
        parsed_model = utils.read_trained_model(output_model, self.settings)

        self.assertEqual(str(type(parsed_model)),
                         "<class 'sklearn.pipeline.Pipeline'>")

    @classmethod
    def tearDownClass(cls):

        for f in cls.model_files:
            # pre-generated models for testing predict are marked with x
            # therefore only remove non-pregenerated ones
            f_path = os.path.join(cls.settings['MODEL_PATH'], f)
            if os.path.isfile(f_path):
                os.unlink(f_path)
        os.unlink('stdout_tmp')

class test_predict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.settings_fh = 'test_settings.json'
        cls.settings = utils.get_settings(cls.settings_fh)
        cls.NULL = open(os.devnull, 'w')
        cls.proc = subprocess.call(['../predict.py',
                                      '-s', 'test_settings.json'],
                                      stdout=cls.NULL,
                                      stderr=cls.NULL)

        cls.output_file = os.listdir(cls.settings['SUBMISSION_PATH'])

    def test_file_output(self):
        '''
        Test whether a file was actually outputted
        '''
        # Check whether there is only one output in submission path
        # and that it is called submission.csv
        self.assertEqual(len(self.output_file), 1)
        self.assertEqual(self.output_file[0], 'submission.csv')

    def test_csv_valid(self):
        '''
        Test whether file is a csv of the right dimensions
        '''
        # parse submission csv into list of lists with csv reader
        with open(os.path.join(self.settings['SUBMISSION_PATH'],
                               self.output_file[0]),
                  'r') as csv_out_file:
            parsed_contents = [row for row \
                    in csv.reader(csv_out_file, delimiter=',')]

        # assert csv has right number of rows
        self.assertEqual(len(parsed_contents), 3936)

        # assert all rows have 2 cols
        for row in parsed_contents:
            self.assertEqual(len(row), 2)

    @classmethod
    def tearDownClass(cls):

        cls.NULL.close()
        for f in cls.output_file:
            # pre-generated models for testing predict are marked with x
            # therefore only remove non-pregenerated ones
            f_path = os.path.join(cls.settings['SUBMISSION_PATH'], f)
            if os.path.isfile(f_path):
                os.unlink(f_path)

if __name__=='__main__':

    unittest.main()
