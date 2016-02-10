hail-seizure
============

Submission for Kaggle's American Epilepsy Society Seizure Prediction Challenge

http://www.kaggle.com/c/seizure-detection

This README and repository modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices

##Hardware / OS platform used
 * Various servers owned by Edinburgh University Informatics Department: 
      - 64 AMD Opteron cores, 256GB RAM, 4TB disk 
      - Scientific Linux
 * Various mid-high end desktops and laptops:
      - Intel processors (i3 and Xeons), 8-64GB RAM, 0.5-8TB disk
      - Arch Linux

##Dependencies

###Required

 * MATLAB or Octave
 * Python 3.4.1
    - scikit_learn-0.15.2
    - numpy-1.8.1
    - scipy
    - h5py

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


Then run `./preprocessing.m` with:
```
matlab -nodisplay -nosplash -r "preprocessing"
```
or similar.

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

To run alternative models the options can be accessed through the standard
help interface:

```
./train.py -h
```

### Cross validation

Cross validation is run in the process of the `train.py` script.
The AUC for each subject and over all subjects is calculated and saved to the 
If the verbose option is set this will also print the calculated values to the command line.

_Important note_: cross validation is run by splitting the data over the hours that it is split into.
This is very important, as this respects the split between training and test data for the leader board.

##Make prediction

After running `train.py` model files will be generated in the default model 
(`model`) directory. These will be automatically loaded along with the test data
to classify the test data points. The results will be written to an output
csv in the default output directory (`output`):

```
./predict.py
```

As above, options can be viewed by:

```
./predict.py -h
```

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
* `THRESHOLD`: if present will activate VarianceThreshold
* `PCA`: if present will activate Principle Component analysis transform, options
not implemented
* `SELECTION`: if present will activate univariate feature selection. Dictionary
inside each of these keys will be used as options, keys are:
    * `KBEST`: Scikit-learn [K-best][kbest] using f-values
    * `PERCENTILE`: Scikit-learn [percentile best][percentile] using f-values
    * `FOREST`: Scikit-learn [Extra-tree transformation][extra]
    * `SVC`: Scikit-learn [linear SVC][lsvc], options are hardcoded
* `TREE_EMBEDDING`: [Random Tree Embedding][rfembed] transformation
* `BAGGING`: [meta-bagger][meta] using selected classifier as base, options are 
set as a dictionary at this key.
* `RFE`: use [recursive feature elimination][rfe], only works with linear SVC

[kbest]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
[percentile]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
[extra]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
[lsvc]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
[rfembed]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html
[meta]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
[rfe]: http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html

## Model documentation

Our final model was a combination of four models, all of which used a support 
vector machine classifier with feature selection. Notes on this, and the code
actually used in the competition can be found the [Comparing outputs][comparing]
IPython notebook. The settings for each of these models can be found in the 
`settings` directory of the repository.

The important part of this code that can combine the outputs to produce the
final csv can be found in the `average.py` script. Calling this with the 
four csvs four csvs found in merge.json will produce our final output csv:

```json
merge.json
----------
["output/forestselection_gavin_submission_using__v2_feats.csv",
 "output/SVC_best_for_each_subject_in_batchall_with_FS_submission_using__v3_feats.csv",
 "output/stoch_opt_2nd_submission_using__v2_feats.csv",
 "output/bbsubj_pg_submission_using__v2_feats.csv"]
```

```
./average.py -s merge.json -o merged_many_v1.csv
```


