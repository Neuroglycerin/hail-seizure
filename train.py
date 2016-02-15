#!/usr/bin/env python3

import python.utils as utils
import os
import joblib
import pickle
import pdb


def main(settingsfname, verbose=False, store_models=True,
         store_features=False, save_training_detailed=False,
         load_pickled=False, parallel=0):

    settings = utils.get_settings(settingsfname)

    utils.print_verbose('=== Settings file   ===', flag=verbose)
    utils.print_verbose(settingsfname, flag=verbose)
    utils.print_verbose('=== Settings loaded ===', flag=verbose)
    utils.print_verbose(settings, flag=verbose)
    utils.print_verbose('=======================', flag=verbose)

    subjects = settings['SUBJECTS']

    data = utils.get_data(settings, verbose=verbose)

    metadata = utils.get_metadata()

    features_that_parsed = [feature for feature in
                            settings['FEATURES'] if feature in list(data.keys())]

    settings['FEATURES'] = features_that_parsed

    if not settings['FEATURES']:
        raise EnvironmentError('No features could be loaded')

    utils.print_verbose("=====Feature HDF5s parsed=====", flag=verbose)

    model_pipe = utils.build_model_pipe(settings)

    utils.print_verbose("=== Model Used ===\n"
                        "{0}\n==================".format(model_pipe),
                        flag=verbose)

    # dictionary to store results
    subject_predictions = {}

    # dictionary to store features in
    transformed_features = {}

    # if we're loading pickled features then load them
    if load_pickled:
        if isinstance(load_pickled, str):
            with open(load_pickled, "rb") as fh:
                Xtra = pickle.load(fh)
        else:
            with open(settingsfname.split(".")[0]
                      + "_feature_dump.pickle", "rb") as fh:
                Xtra = pickle.load(fh)
    else:
        Xtra = None

    # dictionary for final scores
    auc_scores = {}

    if not parallel:
        for subject in subjects:
            utils.print_verbose(
                "=====Training {0} Model=====".format(str(subject)),
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
                                                            load_pickled,
                                                            settingsfname,
                                                            verbose,
                                                            extra_data=Xtra)
                subject_predictions = None
            elif 'CUSTOM' in settings:
                results, auc = utils.train_custom_model(settings,
                                                        data,
                                                        metadata,
                                                        subject,
                                                        model_pipe,
                                                        store_models,
                                                        load_pickled,
                                                        verbose,
                                                        extra_data=Xtra)
                subject_predictions[subject] = results

            else:
                results, auc = utils.train_model(settings,
                                                 data,
                                                 metadata,
                                                 subject,
                                                 model_pipe,
                                                 store_models,
                                                 load_pickled,
                                                 verbose,
                                                 extra_data=Xtra)
                subject_predictions[subject] = results

            auc_scores.update({subject: auc})

    if parallel:
        if 'RFE' in settings:
            raise NotImplementedError('Parallel RFE is not implemented')

        else:
            output = joblib.Parallel(n_jobs=parallel)(
                joblib.delayed(utils.train_model)(settings,
                                                  data,
                                                  metadata,
                                                  subject,
                                                  model_pipe,
                                                  store_models,
                                                  load_pickled,
                                                  verbose,
                                                  extra_data=Xtra,
                                                  parallel=parallel)
                                                      for subject in subjects)

            results = [x[0] for x in output]
            aucs = [x[1] for x in output]

        for result in results:
            subject_predictions.update(result)

        for auc in aucs:
            auc_scores.update(auc)

    if save_training_detailed:
        with open(save_training_detailed, "wb") as fh:
            pickle.dump(subject_predictions[subject], fh)

    combined_auc = utils.combined_auc_score(settings,
                                            auc_scores,
                                            subj_pred=subject_predictions)

    print(
        "predicted AUC score over all subjects: {0:.2f}".format(combined_auc))
    auc_scores.update({'all': combined_auc})
    utils.output_auc_scores(auc_scores, settings)

    return auc_scores

if __name__ == '__main__':

    # get and parse CLI options
    parser = utils.get_parser()
    args = parser.parse_args()

    main(args.settings,
         verbose=args.verbose,
         save_training_detailed=args.pickle_detailed,
         parallel=int(args.parallel))
