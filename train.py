#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json

def get_data(features):
    return {feat_name: utils.parse_matlab_HDF5(feat_name) for feat_name in features}

if __name__=='__main__':

    settings = json.load(open('SETTINGS.json', 'r'))

    features = settings['FEATURES']

    data = get_data(features)

    subjects = settings['SUBJECTS']

    #thresh = utils.get_thresh()

    #selector = utils.get_selector(k=3000)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    #model_pipe = utils.get_model([('thr',thresh),('sel',selector),('scl',scaler),('cls',classifier)])
    model_pipe = utils.get_model([('scl',scaler),('clf',classifier)])

    #dictionary to store results
    subject_predictions = {}

    for subject in subjects:

        X,y = utils.build_training(subject, features, data)

        # this cross-val is broken at the moment, for reasons discussed in the meeting
        cv = utils.get_cross_validation_set(y)

        # initialise lists for cross-val results
        predictions = []
        labels = []
        allweights = []

        # run cross validation and report results
        for train, test in cv:
            # calculate the weights
            weights = utils.get_weights(y[train])
            # fit the model to the training data
            model_pipe.fit(X[train],y[train],clf__sample_weight=weights)
            # append new predictions
            predictions.append(model_pipe.predict_proba(X[test]))
            # append test weights to store (why?)
            weights = utils.get_weights(y[test])
            allweights.append(weights)
            # store true labels
            labels.append(y[test])

        # stack up the results
        predictions = utils.np.vstack(predictions)[:,1]
        labels = utils.np.hstack(labels)
        weights = utils.np.hstack(allweights)

        # calculate the total AUC score
        auc = utils.sklearn.metrics.roc_auc_score(labels,predictions,sample_weight=weights)
        print("predicted AUC score for {1}: {0}".format(auc,subject))

        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        utils.serialise_trained_model(model_pipe, "model_{0}_{1}".format(subject,settings["VERSION"]))

        #store results from each subject
        subject_predictions[subject] = (predictions,labels,weights)

    #stack subject results (don't worrry about this line)
    predictions,labels,weights = map(utils.np.hstack, zip(*list(subject_predictions.values())))

    # calculate the total AUC score over all subjects
    # not using sample_weight here due to error, should probably be fixed
    auc = utils.sklearn.metrics.roc_auc_score(labels,predictions)
    print("predicted AUC score over all subjects: {0}".format(auc))
