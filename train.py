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

    # this cross-val is broken at the moment, for reasons discussed in the meeting
    cv = utils.get_cross_validation_set(y)

    #thresh = utils.get_thresh()

    #selector = utils.get_selector(k=3000)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    #model_pipe = utils.get_model([('thr',thresh),('sel',selector),('scl',scaler),('cls',classifier)])
    model_pipe = utils.get_model([('scl',scaler),('cls',classifier)])

    for subject in subjects:

        X,y = utils.build_training(subjects[0], features, data)

        # initialise lists for cross-val results
        predictions = []
        labels = []
        allweights = []

        # run cross validation and report results
        for (train, test), subject in zip(cv,subjects):
            # calculate the weights
            weights = utils.get_weights(y[train])
            # fit the model to the training data
            model_pipe.fit(X[train],y[train],clf__sample_weight=weights)
            # append new predictions
            predictions.append(model.predict_proba(X[test]))
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
        print("predicted AUC score: {0}".format(auc))

        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        serialise_trained_model(model_pipe, "model_{0}_{1}".format(subject,setting["VERSION"]))
