#!/usr/bin/env python

import unittest
import python.utils as utils
import os
import json

class ParserTest(unittest.TestCase):

    def setUp(self):
        '''
        Find latest hdf5 in grive folder
        Find the SETTINGS.json
        '''
        self.hdf5_file = None
        self.settings_file = 'SETTINGS.json'

    def test_json_parser(self):
        '''
        Test whether json parser works by reading SETTINGS.json
        Then choose a random
        '''
        self.settings_fh = open(self.settings_file, 'r')
        self.settings = json.load(self.settings_fh)
        self.assertIs(type(self.settings), dict)


    def test_hdf5_parsers(self):
        '''
        '''
        #self.parsed_hdf5 = parse_matlab_HDF5(feat, settings=json_settings)
        #self.assertIs(type(self.settings['FEATURES'], list))






if __name__=='__main__':

        unittest.main(verbosity=2, failfast=True)
