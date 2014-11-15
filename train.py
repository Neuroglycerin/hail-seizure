#!/usr/bin/env python3

import python.utils as utils
import os
import pickle
import pdb

def main(settingsfname, verbose=False, store_models=True,
        store_features=False, save_training_detailed=False):

    settings = utils.get_settings(settingsfname)

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

    # dictionary to store features in
    transformed_features = {}

    auc_scores = {}

    for subject in subjects:
        utils.print_verbose("=====Training {0} Model=====".format(str(subject)),
                            flag=verbose)

        if 'RFE' in settings:
            transformed_features, auc = utils.train_RFE(settings,
                                                        data,
                                                        metadata,
                                                        subject,
                                                        model_pipe,
                                                        transformed_features,
                                                        store_models,
                                                        store_features,
                                                        settingsfname,
                                                        verbose)
            subject_predictions = None

        else:
            results, auc = utils.train_model(settings,
                                             data,
                                             metadata,
                                             subject,
                                             model_pipe,
                                             store_models,
                                             verbose)
            subject_predictions[subject] = results

        auc_scores.update({subject: auc})


    if save_training_detailed:
        with open(save_training_detailed, "wb") as fh:
            pickle.dump(subject_predictions[subject], fh)


    combined_auc = utils.combined_auc_score(settings,
                                            auc_scores,
                                            subj_pred=subject_predictions)

    print("predicted AUC score over all subjects: {0:.2f}".format(combined_auc))
    auc_scores.update({'all': combined_auc})

    utils.output_auc_scores(auc_scores, settings)

    return auc_scores

if __name__=='__main__':

    #get and parse CLI options
    parser = utils.get_parser()
    (opts, args) = parser.parse_args()

    main(opts.settings, verbose=opts.verbose, save_training_detailed=opts.pickle_detailed)
