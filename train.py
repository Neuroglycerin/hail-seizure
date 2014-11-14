#!/usr/bin/env python3

import python.utils as utils
import os
import pickle
import pdb

def main(settings, verbose=False, store_models=True, save_training_detailed=False):

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

    #dictionary to store results
    subject_predictions = {}

    auc_scores = {}

    for subject in subjects:
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

        if 'RFE' in settings:
            # first have to transform
            Xt = model_pipe.named_steps['scl'].fit_transform(X)
            if 'thr' in [step[0] for step in model_pipe.steps]:
                Xt = model_pipe.named_steps['thr'].fit_transform(Xt)
            # we might have huge numbers of features, best to remove in large numbers
            stepsize = int(X.shape[1]/20)
            rfecv = utils.sklearn.feature_selection.RFECV(estimator=model_pipe.named_steps['clf'], 
                step=stepsize, cv=cv, **settings['RFE'])
            rfecv.fit(Xt,y)
            # take the best grid score as the max
            auc = max(rfecv.grid_scores_)
        else:
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

        # add AUC scores to a subj dict
        auc_scores.update({subject: auc})

        if store_models:
            # fit the final model
            weights = utils.get_weights(y)
            
            if 'RFE' in settings:
                utils.serialise_trained_model(rfecv,
                                              subject,
                                              settings,
                                              verbose=verbose)
            else:
                # save it
                model_pipe.fit(X,y,clf__sample_weight=weights)
                utils.serialise_trained_model(model_pipe,
                                              subject,
                                              settings,
                                              verbose=verbose)

        if 'RFE' not in settings:
            #store results from each subject
            subject_predictions[subject] = (predictions, labels, weights, segments)


    if 'RFE' in settings:
        auc = utils.np.mean(list(auc_scores.values()))
    else:
        #stack subject results (don't worry about this line)
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

    return auc_scores

if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    main(opts.settings, verbose=opts.verbose, save_training_detailed=opts.pickle_detailed)
