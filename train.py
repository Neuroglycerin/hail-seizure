#!/usr/bin/env python3

import python.utils as utils

def main(opts):

    settings = utils.get_settings(opts.settings)

    subjects = settings['SUBJECTS']

    data = utils.get_data(settings, verbose=opts.verbose)

    metadata = utils.get_metadata()

    features_that_parsed = list(data.keys())

    utils.print_verbose("=====Feature HDF5s parsed=====", flag=opts.verbose)

    #thresh = utils.get_thresh()

    #selector = utils.get_selector(k=opts.selector_k)

    scaler = utils.get_scaler()

    # get settings should convert class name string to actual classifier
    # object
    classifier = settings['CLASSIFIER']

    # dict of classifier options
    classifier_settings = settings['CLASSIFIER_OPTS']

    #utils.sklearn.svm.SVC(probability=True)

    model_pipe = utils.get_model([('scl', scaler),
                                  ('clf', classifier)])
    #dictionary to store results
    subject_predictions = {}

    auc_scores = {}

    for subject in subjects:
        utils.print_verbose("=====Training {0} Model=====".format(str(subject)),
                            flag=opts.verbose)

        # initialise the data assembler
        assembler = utils.DataAssembler(settings, data, metadata)
        X,y = assembler.build_training(subject)

        # get the CV iterator
        cv = utils.Sequence_CV(assembler.training_segments, metadata)

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
        auc = utils.sklearn.metrics.roc_auc_score(labels,
                                                  predictions,
                                                  sample_weight=weights)

        print("predicted AUC score for {1}: {0:.2f}".format(auc, subject))

        # add AUC scores to a subj dict
        auc_scores.update({subject: auc})

        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        utils.serialise_trained_model(model_pipe,
                                      subject,
                                      settings,
                                      verbose=opts.verbose)

        #store results from each subject
        subject_predictions[subject] = (predictions, labels, weights)


    #stack subject results (don't worrry about this line)
    predictions, labels, weights = map(utils.np.hstack,
                                     zip(*list(subject_predictions.values())))

    # calculate the total AUC score over all subjects
    # not using sample_weight here due to error, should probably be fixed
    auc = utils.sklearn.metrics.roc_auc_score(labels, predictions)

    print("predicted AUC score over all subjects: {0:.2f}".format(auc))

    # output AUC scores to file
    auc_scores.update({'all': auc})

    utils.output_auc_scores(auc_scores)

if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    main(opts)
