#!/usr/bin/env python

import python.utils as utils

if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    settings = utils.get_settings(opts.settings)

    features = settings['FEATURES']
    subjects = settings['SUBJECTS']

    data = utils.get_data(features, settings)

    #thresh = utils.get_thresh()

    #selector = utils.get_selector(k=opts.selector_k)

    scaler = utils.get_scaler()

    classifier = utils.get_classifier()

    model_pipe = utils.get_model([('scl', scaler),
                                  ('clf', classifier)])

    # set depth to something lower
    model_pipe.set_params(clf__max_depth=opts.max_depth)

    #dictionary to store results
    subject_predictions = {}

    for subject in subjects:

        X,y,cv,segments = utils.build_training(subject, features, data)

        # initialise lists for cross-val results
        predictions = []
        labels = []
        allweights = []

        # run cross validation and report results
        for train, test in cv:
            # calculate the weights
            weights = utils.get_weights(y[train])
            # fit the model to the training data
            model_pipe.fit(X[train], y[train], clf__sample_weight=weights)
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
        auc = utils.sklearn.metrics.roc_auc_score(\
                labels,
                predictions,
                sample_weight=weights)

        print("predicted AUC score for {1}: {0:.2f}".format(auc,subject))

        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        utils.serialise_trained_model(\
                model_pipe,
                "model_{0}_{1}".format(subject, settings["VERSION"]),
                settings)

        #store results from each subject
        subject_predictions[subject] = (predictions, labels, weights)

    #stack subject results (don't worrry about this line)
    predictions,labels,weights = map(utils.np.hstack,
                                     zip(*list(subject_predictions.values())))

    # calculate the total AUC score over all subjects
    # not using sample_weight here due to error, should probably be fixed
    auc = utils.sklearn.metrics.roc_auc_score(labels,predictions)
    print("predicted AUC score over all subjects: {0:.2f}".format(auc))
