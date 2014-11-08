#!/usr/bin/env python3

import random
import numpy as np
import sklearn.pipeline
import json
import unittest
import glob
import warnings
import subprocess
import os
import python.utils as utils
import csv
import h5py

class testHDF5parsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.settings_fh = 'test_settings.json'
        cls.settings = utils.get_settings(cls.settings_fh)
        cls.all_subjects = set(['Dog_1',
                             'Dog_2',
                             'Dog_3',
                             'Dog_4',
                             'Dog_5',
                             'Patient_1',
                             'Patient_2'])
        cls.all_types = set(['preictal',
                              'MIerr',
                              'test',
                              'MI',
                              'pseudointerictal',
                              'interictal',
                              'pseudopreictal'])

        cls.malformed_feat = 'malformed_feat'
        cls.malformed_file = os.path.join(cls.settings['TRAIN_DATA_PATH'],
                                          "{0}{1}.h5".format(
                                                    cls.malformed_feat,
                                                    cls.settings['VERSION']))

        cls.malformed_feat = h5py.File(cls.malformed_file, 'w')
        cls.malformed_feat.create_dataset('malfie', (10,10))
        cls.malformed_feat.close()

    def test_ioerror_warning(self):
        '''
        Assert a non-existent file correctly raises warning
        '''

        non_existent_feat = 'fake_feat'
        h5_file_name = os.path.join(self.settings['TRAIN_DATA_PATH'],
                                    "{0}{1}.h5".format(non_existent_feat,
                                                       self.settings['VERSION']))

        with warnings.catch_warnings(record=True) as w:
                dummy = utils.parse_matlab_HDF5(non_existent_feat,
                                                self.settings)

                self.assertEqual(len(w), 1, msg="Check there is one and only "
                                                "one warning raised")
                self.assertIs(w[-1].category, UserWarning, msg="Check that "
                                                                "warning raised "
                                                                "is a UserWarning ")
                self.assertEqual(str(w[-1].message),
                                 "{0} does not exist (or is not "
                                 "readable)".format(h5_file_name), msg="Check the "
                                                                        "warning is "
                                                                        "the correct "
                                                                        "format ")

    def test_parse_error_warning(self):
        '''
        Assert malformed HDF5 raises proper warning
        '''


        malformed_feat = 'malformed_feat'
        h5_file_name = os.path.join(self.settings['TRAIN_DATA_PATH'],
                                    "{0}{1}.h5".format(malformed_feat,
                                                       self.settings['VERSION']))

        with warnings.catch_warnings(record=True) as w:

                dummy = utils.parse_matlab_HDF5(malformed_feat,
                                                self.settings)

                self.assertEqual(len(w), 1, msg="Check one and only one error "
                                                "raised")
                self.assertIs(w[-1].category, UserWarning, msg="Check error is "
                                                                "UserWarning")
                self.assertEqual(str(w[-1].message), "Unable to "
                                                     "parse {0}".format(\
                                                                h5_file_name),
                                 msg="Check the warning raised is correct format")


    def test_hdf5_parse(self):
        '''
        Very basic check of parsed object structure containing minimum 3 dicts
        '''
        parsed_HDF5 = utils.parse_matlab_HDF5(self.settings['FEATURES'][0],
                                              self.settings)

        subjects = set(parsed_HDF5.keys())
        self.assertEqual(subjects,
                         self.all_subjects,
                         msg="Check that parsed HDF5 contains all subjects")

        typs = set(parsed_HDF5['Dog_1'].keys())
        self.assertEqual(typs,
                         self.all_types,
                         msg="Check that all ictypes are in parsed file by "
                             "checking the ictypes under Dog_1")

        num_interictal_dog1_segs = len(\
                parsed_HDF5['Dog_1']['interictal'].keys())

        self.assertEqual(num_interictal_dog1_segs,
                         480,
                         msg="Check there is the correct number of segments in "
                             "parsed HDF5s by checking dog1 interictal")

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.malformed_file)

class testTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.settings_fh = 'test_train.json'
        cls.settings = utils.get_settings(cls.settings_fh)

        f = open('stdout_tmp', 'w')
        cls.proc = subprocess.call(['../train.py',
                                      '-s', 'test_train.json'],
                                      stdout=f)
        f.close()

        with open('stdout_tmp', 'r') as f:
            cls.stdout = f.read()

        cls.model_files = glob.glob(os.path.join(cls.settings['MODEL_PATH'],
                                        "{0}_model_for_*_using_{1}_feats.model".format(\
                                                        cls.settings['RUN_NAME'],
                                                        cls.settings['VERSION'])))

    def test_train_stdout(self):
        '''
        Test stdout prints correct number of AUC scores
        '''
        # count the number of AUC scores printed to stdout
        # and assert this is 8 (7 subjects and 1 overall)
        AUC_score_count = self.stdout.count('AUC')
        self.assertEqual(AUC_score_count,
                         8,
                         msg="Check that train prints 8 AUC scores to stdout")

    def test_model_number(self):
        '''
        Test correct number of models are generated
        '''
        # get number of models
        self.assertEqual(len(self.model_files),
                         7,
                         msg="Check that 7 models are written out to model_path dir")

    def test_model_size_correct(self):
        '''
        Test if one of the serialised models is roughly correct size
        '''
        # randomly pick an output model
        output_model = random.choice(self.model_files)

        # get file size and assert between 2.5 and 8k
        output_model_stats = os.stat(output_model)
        output_model_size = output_model_stats.st_size
        self.assertTrue(1000 < output_model_size < 20000,
                        msg="Check that randomly picked model ({0}) is between 1 "
                            "and 10M".format(output_model))

    def test_model_can_be_read(self):
        '''
        Check whether a model can be read
        '''
        output_model = random.choice(self.settings['SUBJECTS'])
        parsed_model = utils.read_trained_model(output_model, self.settings)

        self.assertIsInstance(parsed_model,
                              sklearn.pipeline.Pipeline,
                              msg="Check that randomly picked model ({0}) is "
                                  "the correct sklearn obj type".format(output_model))

    @classmethod
    def tearDownClass(cls):
        '''
        Remove generated model files and stdout output temp file
        '''
        for f in cls.model_files:
            os.unlink(f)

        os.unlink('stdout_tmp')


class testPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.settings_fh = 'test_predict.json'
        cls.settings = utils.get_settings(cls.settings_fh)
        cls.NULL = open(os.devnull, 'w')
        cls.proc = subprocess.call(['../predict.py',
                                      '-s', 'test_predict.json'],
                                      stdout=cls.NULL,
                                      stderr=cls.NULL)


        cls.output_file = glob.glob(os.path.join(cls.settings['SUBMISSION_PATH'],
                                    "*.csv"))


    def test_file_output(self):
        '''
        Test whether a file was actually outputted
        '''
        # Check whether there is only one output in submission path
        self.assertEqual(len(self.output_file), 1, msg="Check only one csv is "
                                                       "output to output path")
        self.assertEqual(self.output_file[0],
                         os.path.join(self.settings['SUBMISSION_PATH'],
                                      '{0}_submission_using_{1}_feats'
                                      '.csv'.format(self.settings['RUN_NAME'],
                                                    self.settings['VERSION'])),
                         msg="Checking that the output csv has the right "
                             "abspath and filename")

    def test_csv_valid(self):
        '''
        Test whether file is a csv of the right dimensions
        '''
        # parse submission csv into list of lists with csv reader
        with open(self.output_file[0], 'r') as csv_out_file:
            parsed_contents = [row for row \
                    in csv.reader(csv_out_file, delimiter=',')]

        # assert csv has right number of rows
        self.assertEqual(len(parsed_contents),
                         3936,
                         msg="Check that output csv has 3936 rows "
                             "(3935 test segments + header)")

        # assert all rows have 2 cols
        for row in parsed_contents:
            self.assertEqual(len(row),
                             2,
                             msg="Check that output csv only has 2 cols")

    @classmethod
    def tearDownClass(cls):
        '''
        Close /dev/null filehandle and remove any csv output
        '''

        cls.NULL.close()
        for f in cls.output_file:
            if f!='.placeholder':
                os.unlink(f)

class testDataAssembler(unittest.TestCase):
    '''
    Unittests for DataAssembler object
    '''
    @classmethod
    def setUpClass(cls):
        cls.settings_fh = 'test_data_assembler.json'
        cls.settings = utils.get_settings(cls.settings_fh)
        cls.subjects = cls.settings['SUBJECTS']
        cls.features = cls.settings['FEATURES']
        cls.data = utils.get_data(cls.settings)
        with open('test_segmentMetadata.json', 'r') as f:
            cls.metadata = json.load(f)

        cls.ictyps = cls.settings['DATA_TYPES']

        cls.segment_counts = {'Dog_1': {'preictal': 24,
                                        'pseudopreictal': 24,
                                        'interictal': 480,
                                        'pseudointerictal': 480,
                                        'test': 502},
                              'Dog_2': {'preictal': 42,
                                        'pseudopreictal': 42,
                                        'interictal': 500,
                                        'pseudointerictal': 500,
                                        'test': 1000},
                              'Dog_3': {'preictal': 72,
                                        'pseudopreictal': 72,
                                        'interictal': 1440,
                                        'pseudointerictal': 1440,
                                        'test': 907},
                              'Dog_4': {'preictal': 97,
                                        'pseudopreictal': 97,
                                        'interictal': 804,
                                        'pseudointerictal': 804,
                                        'test': 990},
                              'Dog_5': {'preictal': 30,
                                        'pseudopreictal': 30,
                                        'interictal': 450,
                                        'pseudointerictal': 450,
                                        'test': 191},
                              'Patient_1': {'preictal': 18,
                                            'pseudopreictal': 18,
                                            'interictal': 50,
                                            'pseudointerictal': 50,
                                            'test': 195},
                              'Patient_2': {'preictal': 18,
                                            'pseudopreictal': 18,
                                            'interictal': 42,
                                            'pseudointerictal': 42,
                                            'test': 150}}
        cls.feature_length = {'Dog_1': 16,
                              'Dog_2': 16,
                              'Dog_3': 16,
                              'Dog_4': 16,
                              'Dog_5': 15,
                              'Patient_1': 15,
                              'Patient_2': 24}
        cls.ictyp_mapping = {'preictal': 1,
                             'interictal': 0}


    def setUp(self):
        self.DataAssemblerInstance = utils.DataAssembler(self.settings,
                                                        self.data,
                                                        self.metadata)


    def test_build_test(self):

        for subj in self.subjects:
            X = self.DataAssemblerInstance.build_test(subj)
            target_X_shape = (self.segment_counts[subj]['test'],
                              self.feature_length[subj] * len(self.features))

            self.assertIsInstance(X,
                                  np.ndarray,
                                  msg="Check that for subj {0} "
                                      "X is an array".format(subj))


            self.assertEqual(X.shape, target_X_shape,
                             msg="Check that for subj {0} "
                                 "X is an right shape".format(subj))


            self.assertTrue(X[:,
                              :self.feature_length[subj]].all() == 0,
                        msg="Check that for subj {0} "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj))

            self.assertTrue(X[:,
                          self.feature_length[subj]:\
                          self.feature_length[subj]*2].all() == 1,
                        msg="Check that for subj {0} "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj))


    def test_build_training(self):

        for subj in self.subjects:
            X, y = self.DataAssemblerInstance.build_training(subj)

            self.assertIsInstance(X,
                                  np.ndarray,
                                  msg="Check that for subj {0} "
                                      "X is an array".format(subj))


            target_X_shape = (self.segment_counts[subj]['interictal'] +
                              self.segment_counts[subj]['preictal'],
                              self.feature_length[subj] * len(self.features))

            self.assertEqual(X.shape, target_X_shape,
                             msg="Check that for subj {0} "
                                 "X is an right shape".format(subj))

            self.assertTrue(X[:,
                              :self.feature_length[subj]].all() == 0,
                        msg="Check that for subj {0} "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj))

            self.assertTrue(X[:,
                          self.feature_length[subj]:\
                          self.feature_length[subj]*2].all() == 1,
                        msg="Check that for subj {0} "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj))

            self.assertIsInstance(y,
                                  np.ndarray,
                                  msg="Check that for subj {0} "
                                      "X is an array".format(subj))

            target_y_shape = (self.segment_counts[subj]['interictal'] +
                              self.segment_counts[subj]['preictal'], )

            self.assertEqual(y.shape, target_y_shape,
                             msg="Check that for subj {0} "
                                 "y is an right shape".format(subj))



    def test__build_y(self):
        '''
        For each subj and ictyp make sure the y vector returned is the right
        size and has the right values
        '''
        for subj in self.subjects:
            for ictyp in ['preictal', 'interictal']:
                y = self.DataAssemblerInstance._build_y(subj, ictyp)
                self.assertIsInstance(y,
                                      np.ndarray,
                                      msg="Check that y for subj {0} and "
                                          "icty {1} is numpy array".format(subj,
                                                                           ictyp))
                self.assertEqual(y.shape[0],
                                 self.segment_counts[subj][ictyp],
                                 msg="Check that y is right length for subj {0} and "
                                     "icty {1} is numpy array".format(subj, ictyp))

                self.assertTrue(all(y == self.ictyp_mapping[ictyp]),
                                msg="Check that y is all right value for subj {0} and "
                                     "icty {1} is numpy array".format(subj, ictyp))


    def test__build_y_error_on_test(self):
        '''
        Test whether y throws error if you attempt to create vector with the
        test data ictyp
        '''
        subj = random.choice(self.subjects)
        self.assertRaises(ValueError,
                          self.DataAssemblerInstance._build_y,
                          subj,
                          'test')

    def test__build_X(self):
        '''
        Test _build_x is correctly returning the right shaped
        matrix for every subj and ictyp using our fixed
        data
        '''
        for subj in self.subjects:
            for ictyp in self.ictyps:
                X, index = self.DataAssemblerInstance._build_X(subj, ictyp)
                self.assertIsInstance(X,
                                      np.ndarray,
                                      msg="Check that for subj {0} and ictyp {1} "
                                          "X is an array".format(subj, ictyp))
                self.assertEqual(X.shape,
                                (self.segment_counts[subj][ictyp],
                                    self.feature_length[subj]*2),
                                 msg="Check that for subj {0} and ictyp {1} "
                                     "X is an right shape".format(subj,
                                                                  ictyp))
    def test__build_X_ordering(self):
        '''
        Check order of the features is preserved on assembly
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)
        X, index = self.DataAssemblerInstance._build_X(subj, ictyp)

        self.assertTrue(X[:,
                          :self.feature_length[subj]].all() == 0,
                        msg="Check that for random subj {0} and ictyp {1} "
                            "when reading an all 0 then all 1 feature in order "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj, ictyp))
        self.assertTrue(X[:,
                          self.feature_length[subj]:\
                          self.feature_length[subj]*2].all() == 1,
                        msg="Check that for random subj {0} and ictyp {1} "
                            "when reading an all 0 then all 1 feature in order "
                            "X is generated with 0 first then 1 "
                            "afterwards".format(subj, ictyp))




    def test__build_X_feature_index(self):
        '''
        Check feature index is correctly made
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)
        X, feature_index = self.DataAssemblerInstance._build_X(subj, ictyp)
        self.assertEqual((feature_index[0], feature_index[-1]),
                         self.features,
                         msg="Check that for random subj {0} and ictyp {1} "
                             "feature index is same order as features "
                             "are in settings".format(subj, ictyp))

    def test__assemble_feature(self):
        '''
        Check assembly of random ictype or feature
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)
        feature = random.choice(self.features)

        X_part = self.DataAssemblerInstance._assemble_feature(subj,
                                                              feature,
                                                              ictyp)
        self.assertIsInstance(X_part,
                              np.ndarray,
                              msg="Check that for random subj {0} and ictyp {1} "
                             "X_part is an array".format(subj, ictyp))

        self.assertEqual(X_part.shape, (self.segment_counts[subj][ictyp],
                                        self.feature_length[subj]),
                         msg="Check that for random subj {0} and ictyp {1} "
                             "X_part is correct shape".format(subj, ictyp))


    def test__parse_segment_names(self):
        '''
        Check parse segment names
        - is a dict containing all the subjects
        - For a random subject:
          * it contains a dict
          * a dict of a size equal to the summed segment count for all ictyps
            of that subject
        '''

        segment_names = self.DataAssemblerInstance._parse_segment_names()
        self.assertEqual(set(segment_names.keys()), set(self.subjects))

        subj = random.choice(self.subjects)
        self.assertIsInstance(segment_names[subj], dict)

        subj_segment_number = sum([self.segment_counts[subj][ictyp] \
                                        for ictyp in self.ictyps])

        length_of_class_segment_names = sum([len(segment_names[subj][ictyp]) \
                                                for ictyp in self.ictyps])

        self.assertEqual(length_of_class_segment_names, subj_segment_number,
                         msg="{0} subj used".format(subj))



    def test_init(self):
        '''
        Test that the class initialises correctly
        '''
        self.assertEqual(self.DataAssemblerInstance.settings, self.settings)
        self.assertEqual(self.DataAssemblerInstance.data, self.data)
        self.assertEqual(self.DataAssemblerInstance.metadata, self.metadata)

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

if __name__=='__main__':

    unittest.main()
