hail-seizure
============

Submission for Kaggle's American Epilepsy Society Seizure Prediction Challenge

http://www.kaggle.com/c/seizure-detection

This README and repository modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices

##Hardware / OS platform used
 * Salmon (server owned by Edinburgh University Informatics Department): 
      - 64 AMD Opteron cores, 256GB RAM, 4TB disk 
      - Scientific Linux
 * Various mid-high end desktops and laptops:
      - Intel processors (i3 and Xeons), 8-64GB RAM, 0.5-8TB disk
      - Arch Linux

##Dependencies

###Required

 * Python 2.7
 * scikit_learn-0.14.1
 * numpy-1.8.1
 * pandas-0.14.0
 * scipy
 * h5py
 * json

##Generate features

Place path to raw data organised by subject under the RAW_DATA_DIRS key of SETTINGS.json and check the values used in the SETTINGS.json
```
RAW_DATA_DIR/
  Dog_1/
    Dog_1_ictal_segment_1.mat
    Dog_1_ictal_segment_2.mat
    ...
    Dog_1_interictal_segment_1.mat
    Dog_1_interictal_segment_2.mat
    ...
    Dog_1_test_segment_1.mat
    Dog_1_test_segment_2.mat
    ...

  Dog_2/
  ...
```


Then run:
```
./preprocessing.m
```

This will calculate features used the feature functions specified in SETTINGS.json FEATURES field and output them to TRAIN_DATA_PATH directory as HDF5 files.

HDF5 structure: 

```
$feature_name.h5 = {$subject: {$type : {$segment_file_name : $feature_vector } } }

```

* `$feature_name.h5`: is the feature name, modification type and version number e.g. (raw_feat_var_v2.h5 or ica_feat_covar_v5.h5 etc)
* `$type`: data type e.g. 'preictal', 'interictal' or 'test'
* `$segment_file_name`: the filename for the segment from which that vector was generated
* `$feature_vector`: A 1xNxM feature vector for that segment using the specified feature function

##Train classifier

One classifier is trained for each patient and serialised into the directory specific in SETTINGS.json under MODEL_PATH (default is model/).

This is achieved by running:
```
./train.py
```

##Cross validation

TO BE DONE

##Make prediction

Run
```
./predict.py
```

TO BE DONE


## SETTINGS.json

```
{
    "TRAIN_DATA_PATH": "train", 
    "MODEL_PATH": "model", 
    "SUBJECTS": ["Dog_1",
                 "Dog_2",
                 "Dog_3",
                 "Dog_4",
                 "Dog_5",
                 "Patient_1",
                 "Patient_2"],
    "FEATURES": ["feat_var",
                 "feat_var", 
                 "feat_cov", 
                 "feat_corrcoef",
                 "feat_pib", 
                 "feat_xcorr", 
                 "feat_psd", 
                 "feat_psd_logf",
                 "feat_coher",
                 "feat_coher_logf"],
    "TEST_DATA_PATH": "test", 
    "SUBMISSION_PATH": "output",
    "VERSION": "_v1",
    "RAW_DATA_DIRS": ["/disk/data2/neuroglycerin/hail-seizure-data/",
                      "/media/SPARROWHAWK/neuroglycerin/hail-seizure-data/",
                      "/media/scott/SPARROWHAWK/neuroglycerin/hail-seizure-data/"]
}
```
* `SUBJECTS`: list of which subjects to use in the current run
* `VERSION`: string to indicate version number of this run
* `RAW_DATA_DIRS`: directory that contains the raw .mat data organised by subject
* `FEATURES`: list of features used in this run
* `TRAIN_DATA_PATH`: directory holding the preprocessed extracted features from raw data in per-feature HDF5s 
* `MODEL_PATH`: directory containing the serialised miodels
* `TEST_DATA_PATH`: directory containing all output related to model testing (CV etc).
* `SUBMISSION_PATH`: directory containing the submission csv for the current run


## Model documentation

TO BE COMPLETED
