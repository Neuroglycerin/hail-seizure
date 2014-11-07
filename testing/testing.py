#!/usr/bin/env python3

import random
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
        cls.malformed_file = "{0}/{1}{2}.h5".format(cls.settings['TRAIN_DATA_PATH'],
                                                    cls.malformed_feat,
                                                    cls.settings['VERSION'])

        cls.malformed_feat = h5py.File(cls.malformed_file, 'w')
        cls.malformed_feat.create_dataset('malfie', (10,10))
        cls.malformed_feat.close()

    def test_ioerror_warning(self):
        '''
        Assert a non-existent file correctly raises warning
        '''

        non_existent_feat = 'fake_feat'
        h5_file_name = "{0}/{1}{2}.h5".format(self.settings['TRAIN_DATA_PATH'],
                                              non_existent_feat,
                                              self.settings['VERSION'])

        with warnings.catch_warnings(record=True) as w:
                dummy = utils.parse_matlab_HDF5(non_existent_feat,
                                                self.settings)

                self.assertEqual(len(w), 1)
                self.assertIs(w[-1].category, UserWarning)
                self.assertEqual(str(w[-1].message),
                                 "{0} does not exist (or is not "
                                 "readable)".format(h5_file_name))

    def test_parse_error_warning(self):
        '''
        Assert malformed HDF5 raises proper warning
        '''


        malformed_feat = 'malformed_feat'
        h5_file_name = "{0}/{1}{2}.h5".format(self.settings['TRAIN_DATA_PATH'],
                                              malformed_feat,
                                              self.settings['VERSION'])

        with warnings.catch_warnings(record=True) as w:

                dummy = utils.parse_matlab_HDF5(malformed_feat,
                                                self.settings)

                self.assertEqual(len(w), 1)
                self.assertIs(w[-1].category, UserWarning)
                self.assertEqual(str(w[-1].message), "Unable to "
                                                     "parse {0}".format(\
                                                                h5_file_name))


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

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.malformed_file)

class testTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.settings_fh = 'test_train.json'
        cls.settings = utils.get_settings(cls.settings_fh)

        f = open('stdout_tmp', 'w')
        cls.proc = subprocess.call(['../train.py', '-t', '1',
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
        output_model_stats = os.stat(output_model)
        output_model_size = output_model_stats.st_size
        self.assertTrue(1000 < output_model_size < 10000)

    def test_model_can_be_read(self):
        '''
        Check whether a model can be read
        '''
        output_model = random.choice(self.settings['SUBJECTS'])
        parsed_model = utils.read_trained_model(output_model, self.settings)

        self.assertEqual(str(type(parsed_model)),
                         "<class 'sklearn.pipeline.Pipeline'>")

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
        self.assertEqual(len(self.output_file), 1)
        self.assertEqual(self.output_file[0],
                         os.path.join(self.settings['SUBMISSION_PATH'],
                                      '{0}_submission_using_{1}_feats'
                                      '.csv'.format(self.settings['RUN_NAME'],
                                                    self.settings['VERSION'])))

    def test_csv_valid(self):
        '''
        Test whether file is a csv of the right dimensions
        '''
        # parse submission csv into list of lists with csv reader
        with open(self.output_file[0], 'r') as csv_out_file:
            parsed_contents = [row for row \
                    in csv.reader(csv_out_file, delimiter=',')]

        # assert csv has right number of rows
        self.assertEqual(len(parsed_contents), 3936)

        # assert all rows have 2 cols
        for row in parsed_contents:
            self.assertEqual(len(row), 2)

    @classmethod
    def tearDownClass(cls):
        '''
        Close /dev/null filehandle and remove any csv output
        '''

        cls.NULL.close()
        for f in cls.output_file:
            if f!='.placeholder':
                os.unlink(f)

class testData_assembler(unittest.TestCase):
    '''
    Unittests for data_assembler object
    '''
    @classmethod
    def setUpClass(cls):
        cls.settings_fh = 'test_data_assembler.json'
        cls.settings = utils.get_settings(cls.settings_fh)
        cls.subjects = cls.settings['SUBJECTS']
        cls.data = utils.get_data(cls.settings)
        cls.metadata = json.load(open('test_segmentMetadata.json',
                                      'r'))
        cls.ictyps = ['preictal', 'interictal', 'test']

        cls.segment_counts = {'Dog_1': {'preictal': 24,
                                        'interictal': 480,
                                        'test': 502},
                              'Dog_2': {'preictal': 42,
                                        'interictal': 500,
                                        'test': 1000},
                              'Dog_3': {'preictal': 72,
                                        'interictal': 1440,
                                        'test': 907},
                              'Dog_4': {'preictal': 97,
                                        'interictal': 804,
                                        'test': 990},
                              'Dog_5': {'preictal': 30,
                                        'interictal': 450,
                                        'test': 191},
                              'Patient_1': {'preictal': 18,
                                            'interictal': 50,
                                            'test': 195},
                              'Patient_2': {'preictal': 18,
                                            'interictal': 42,
                                            'test': 150}}
        cls.feature_length = {'Dog_1': 16,
                              'Dog_2': 16,
                              'Dog_3': 16,
                              'Dog_4': 16,
                              'Dog_5': 15,
                              'Patient_1': 15,
                              'Patient_2': 24}


    def setUp(self):
        self.data_assembler = utils.data_assembler(self.settings,
                                                   self.data,
                                                   self.metadata)


    def test_build_test(self):
        self.data_assembler.build_test()
        pass

    def test_build_training(self):
        self.data_assembler.build_training()
        pass


    def test__build_y(self):
        self.data_assember._build_y()
        pass


    def test__build_X(self):
        '''
        Test _build_x is correctly returning the right shaped
        matrix for every subj and ictyp using our fixed
        data
        '''
        for subj in self.subjects:
            for ictyp in self.ictyps:
                X, index = self.data_assembler._build_X(subj, ictyp)
                self.assertEqual(type(X), 'numpy.ndarray')
                self.assertEqual(X.shape, (self.segment_counts[subj][ictyp],
                                           self.feature_lengh[subj]*2))
    def test__build_X_ordering(self):
        '''
        Check order of the features is preserved on assembly
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)
        X, feature_index = self.data_assembler._build_X(subj, ictyp)

        self.assertAlmostEqual(0, X_part[:,:self.feature_length[subj]])
        self.assertAlmostEqual(1, X_part[:,self.feature_length[subj]:self.feature_length[subj]*2])


    def test__build_X_feature_index(self):
        '''
        Check feature index is correctly made
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)
        X, feature_index = self.data_assembler._build_X(subj, ictyp)
        self.assertEqual(feature_index, self.settings['FEATURES'])


    def test__assemble_feature(self):
        '''
        Check assembly of random ictype or feature
        '''
        ictyp = random.choice(self.ictyps)
        subj = random.choice(self.subjects)

        X_part = self.data_assembler._assemble_feature(subj, ictyp)
        self.assertEqual(type(X_part), 'numpy.ndarray')
        self.assertEqual(X_part.shape, (self.segment_counts[subj][ictyp],
                                        self.feature_length[subj]))

    def test_init(self):
        '''
        Test that the class initialises correctly
        '''
        self.assertEqual(self.data_assembler.settings, self.settings)
        self.assertEqual(self.data_assembler.data, self.data)
        self.assertEqual(self.data_assembler.metadata, self.metadata)

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

if __name__=='__main__':

    unittest.main()
