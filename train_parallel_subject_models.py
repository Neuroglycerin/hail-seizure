#!/usr/bin/env python3

import python.utils as utils
import os
import joblib
import pickle

def main(settings, cores=7, verbose=False, store_models=True, save_training_detailed=False):

    settings = utils.get_settings(settings)

    subjects = settings['SUBJECTS']

    data = utils.get_data(settings, verbose=verbose)

    metadata = utils.get_metadata()

    features_that_parsed = [feature for feature in \
            settings['FEATURES'] if feature in list(data.keys())]

    settings['FEATURES'] = features_that_parsed

    utils.print_verbose("=====Feature HDF5s parsed=====", flag=verbose)

    model_pipe = utils.build_model_pipe(settings)

    utils.print_verbose("=== Model Used ===\n"
    "{0}\n==================".format(model_pipe),
                        flag=verbose)


    results = joblib.Parallel(n_jobs=cores)(\
            joblib.delayed(\
            process_subject_model)(subject,
                                   settings,
                                   data,
                                   metadata,
                                   verbose,
                                   model_pipe,
                                   store_models) for subject in subjects)

    subject_predictions = {}
    auc_scores = {}

    for result in results:
        auc_scores.update(result[0])
        subject_predictions.update(result[1])

    # #stack subject results (don't worrry about this line)
    predictions, labels, weights, segments = map(utils.np.hstack,
                                     zip(*list(subject_predictions.values())))

    if save_training_detailed:
        with open(save_training_detailed, "wb") as fh:
            pickle.dump((predictions, labels, weights, segments), fh)

    # calculate the total AUC score over all subjects
    # not using sample_weight here due to error, should probably be fixed
    auc = utils.sklearn.metrics.roc_auc_score(labels, predictions)

    print("predicted AUC score over all subjects: {0:.2f}".format(auc))

    # output AUC scores to file
    auc_scores.update({'all': auc})

    utils.output_auc_scores(auc_scores, settings)



def process_subject_model(subject,
                          settings,
                          data,
                          metadata,
                          verbose,
                          model_pipe,
                          store_models):

    utils.print_verbose("=====Training {0} Model=====".format(str(subject)),
                        flag=verbose)

    # initialise the data assembler
    assembler = utils.DataAssembler(settings, data, metadata)
    X,y = assembler.build_training(subject)


    # get the CV iterator
    cv = utils.Sequence_CV(assembler.training_segments,
                           metadata,
                           r_seed=settings['R_SEED'],
                           n_iter=settings['CVITERCOUNT'])

    # initialise lists for cross-val results
    predictions = []
    labels = []
    allweights = []
    segments = []

    # run cross validation and report results
    for train, test in cv:

        # calculate the weights
        weights = utils.get_weights(y[train])
        # fit the model to the training data
        model_pipe.fit(X[train], y[train], clf__sample_weight=weights)
        # append new predictions
        predictions.append(model_pipe.predict_proba(X[test]))
        # append test weights to store (why?) (used to calculate auc below)
        weights = utils.get_weights(y[test])
        allweights.append(weights)
        # store true labels
        labels.append(y[test])
        # store segments
        segments.append(assembler.training_segments[test])

    # stack up the results
    predictions = utils.np.vstack(predictions)[:,1]
    labels = utils.np.hstack(labels)
    weights = utils.np.hstack(allweights)
    segments = utils.np.hstack(segments)

    # calculate the total AUC score
    auc = utils.sklearn.metrics.roc_auc_score(labels,
                                              predictions,
                                              sample_weight=weights)

    print("predicted AUC score for {1}: {0:.2f}".format(auc, subject))

    if store_models:
        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        utils.serialise_trained_model(model_pipe,
                                      subject,
                                      settings,
                                      verbose=verbose)

    #store results from each subject
    return ({subject: auc}, {subject: (predictions, labels, weights, segments)})






if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    # change to opts.cores
    main(opts.settings, cores=7, verbose=opts.verbose, save_training_detailed=opts.pickle_detailed)
