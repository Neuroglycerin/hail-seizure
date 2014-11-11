#!/usr/bin/env python3

import python.utils as utils
import os
import pickle

def main(opts):

    settings = utils.get_settings(opts.settings)

    subjects = settings['SUBJECTS']

    data = utils.get_data(settings, verbose=opts.verbose)

    metadata = utils.get_metadata()

    features_that_parsed = [feature for feature in \
            settings['FEATURES'] if feature in list(data.keys())]

    settings['FEATURES'] = features_that_parsed

    utils.print_verbose("=====Feature HDF5s parsed=====", flag=opts.verbose)

    elements = []

    scaler = utils.get_scaler()
    elements.append(('scl',scaler))

    if 'THRESHOLD' in settings.keys():
        thresh = utils.get_thresh()
        elements.append(('thr',thresh))

    if 'SELECTION' in settings.keys():
        selector = utils.get_selector(settings)
        elements.append(('sel',selector))

    # get settings should convert class name string to actual classifier
    # object
    classifier = settings['CLASSIFIER']
    elements.append(('clf',classifier))

    # dict of classifier options - not used yet?
    classifier_settings = settings['CLASSIFIER_OPTS']

    #utils.sklearn.svm.SVC(probability=True)

    model_pipe = utils.get_model(elements)
    utils.print_verbose("=== Model Used ===\n"
    "{0}\n==================".format(model_pipe),flag=opts.verbose)

    #dictionary to store results
    subject_predictions = {}

    accuracy_scores = {}
    KL_scores = {}

    for subject in subjects:
        utils.print_verbose("=====Training {0} Model=====".format(str(subject)),
                            flag=opts.verbose)

        # initialise the data assembler
        assembler = utils.DataAssembler(settings, data, metadata)
        X,y = assembler.test_train_discrimination(subject)

        # get the CV iterator
        cv = utils.sklearn.cross_validation.StratifiedShuffleSplit( \
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

        # calculate the total AUC score
        accuracy = utils.sklearn.metrics.accuracy_score(labels,
                                                  predictions,
                                                  sample_weight=weights)

        print("Accuracy score for {1}: {0:.3f}".format(accuracy, subject))

        # add AUC scores to a subj dict
        accuracy_scores.update({subject: accuracy})

        # fit the final model
        weights = utils.get_weights(y)

        # save it
        model_pipe.fit(X,y,clf__sample_weight=weights)
        utils.serialise_trained_model(model_pipe,
                                      subject,
                                      settings,
                                      verbose=opts.verbose)

        # store results from each subject
        subject_predictions[subject] = (predictions, labels, weights)

        # calculate KL divergence for this dataset
        # first calculate parameters of multivariate Gaussian
        Sigma_train = utils.np.cov(assembler.Xtrain.T)
        Sigma_test = utils.np.cov(assembler.Xtest.T)
        mu_train = utils.np.mean(assembler.Xtrain,axis=0)[utils.np.newaxis].T
        mu_test = utils.np.mean(assembler.Xtest,axis=0)[utils.np.newaxis].T
        
        # calculate KL divergence between these two Gaussians and store
        KL = utils.mvnormalKL(mu_train,mu_test,Sigma_train,Sigma_test)
        KL_scores.update({subject: KL})

        print("KL divergence between"
            " training and test for {1}: {0:.3f}".format(KL,subject))

    #stack subject results (don't worrry about this line)
    predictions, labels, weights = map(utils.np.hstack,
                                     zip(*list(subject_predictions.values())))

    # calculate the total AUC score over all subjects
    # not using sample_weight here due to error, should probably be fixed
    accuracy = utils.sklearn.metrics.accuracy_score(labels, predictions)

    print("predicted accuracy score over all subjects: {0:.2f}".format(accuracy))

    # output AUC scores to file
    accuracy_scores.update({'all': accuracy})

    settings['DISCRIMINATE'] = 'accuracy_scores.csv'
    settings['AUC_SCORE_PATH'] = 'discriminate_scores'
    utils.output_auc_scores(accuracy_scores, settings)
    settings['DISCRIMINATE'] = 'accuracy_scores.csv'
    utils.output_auc_scores(KL_scores, settings)

if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    main(opts)
