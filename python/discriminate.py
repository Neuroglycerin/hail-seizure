#!/usr/bin/env python3

import os
import pickle

import utils


def main(settingsfname, verbose=False):

    settings = utils.get_settings(settingsfname)

    subjects = settings['SUBJECTS']

    data = utils.get_data(settings, verbose=verbose)

    metadata = utils.get_metadata()

    features_that_parsed = [feature for feature in
                            settings['FEATURES'] if feature in list(data.keys())]

    settings['FEATURES'] = features_that_parsed

    utils.print_verbose("=====Feature HDF5s parsed=====", flag=verbose)

    # get model
    model_pipe = utils.build_model_pipe(settings)

    utils.print_verbose("=== Model Used ===\n"
                        "{0}\n==================".format(model_pipe), flag=verbose)

    # dictionary to store results
    subject_predictions = {}

    accuracy_scores = {}

    for subject in subjects:
        utils.print_verbose(
            "=====Training {0} Model=====".format(str(subject)),
                            flag=verbose)

        # initialise the data assembler
        assembler = utils.DataAssembler(settings, data, metadata)
        X, y = assembler.test_train_discrimination(subject)

        # get the CV iterator
        cv = utils.sklearn.cross_validation.StratifiedShuffleSplit(
            y,
                               random_state=settings['R_SEED'],
                               n_iter=settings['CVITERCOUNT'])

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
            predictions.append(model_pipe.predict(X[test]))
            # append test weights to store (why?) (used to calculate auc below)
            weights = utils.get_weights(y[test])
            allweights.append(weights)
            # store true labels
            labels.append(y[test])

        # stack up the results
        predictions = utils.np.hstack(predictions)
        labels = utils.np.hstack(labels)
        weights = utils.np.hstack(allweights)

        # calculate the total accuracy
        accuracy = utils.sklearn.metrics.accuracy_score(labels,
                                                        predictions,
                                                        sample_weight=weights)

        print("Accuracy score for {1}: {0:.3f}".format(accuracy, subject))

        # add AUC scores to a subj dict
        accuracy_scores.update({subject: accuracy})

        # store results from each subject
        subject_predictions[subject] = (predictions, labels, weights)

    # stack subject results (don't worrry about this line)
    predictions, labels, weights = map(utils.np.hstack,
                                       zip(*list(subject_predictions.values())))

    # calculate global accuracy
    accuracy = utils.sklearn.metrics.accuracy_score(labels, predictions,
                                                    sample_weight=weights)

    print(
        "predicted accuracy score over all subjects: {0:.2f}".format(accuracy))

    # output AUC scores to file
    accuracy_scores.update({'all': accuracy})

    settings['DISCRIMINATE'] = 'accuracy_scores.csv'
    # settings['AUC_SCORE_PATH'] = 'discriminate_scores'
    utils.output_auc_scores(accuracy_scores, settings)

    return accuracy_scores

if __name__ == '__main__':

    # get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    main(opts)
