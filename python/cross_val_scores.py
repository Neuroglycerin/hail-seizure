#!/usr/bin/env python3

import numpy as np
import pickle
import sklearn.metrics
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.svm

import utils


def main():
    metadata = utils.get_metadata()
    settings = utils.get_settings('probablygood.gavin.json')
    settings['R_SEED'] = None
    # settings['SUBJECTS'] = ['Patient_2']
    scaler = sklearn.preprocessing.StandardScaler()
    thresh = sklearn.feature_selection.VarianceThreshold()
    # selector = sklearn.feature_selection.SelectKBest()
    classifier = sklearn.svm.SVC(probability=True)
    pipe = sklearn.pipeline.Pipeline([('scl', scaler),
                                      ('thr', thresh),
                                      #                                  ('sel', selector),
                                      ('cls', classifier)])

    output = {}

    data = utils.get_data(settings)
    da = utils.DataAssembler(settings, data, metadata)
    global_results = {}
    for subject in list(settings['SUBJECTS']) + ['global']:
        global_results[subject] = {}

    for i in range(10):
        print("iteration {0}".format(i))

        for subject in settings['SUBJECTS']:
            print(subject)
            X, y = da.build_training(subject)
            # cv = utils.Sequence_CV(da.training_segments, metadata)
            train, test, train_results, test_results = fit_and_return_parts_and_results(
                da,
                                                                    metadata,
                                                                    pipe,
                                                                    X,
                                                                    y)
            output.update({subject: {'train': train,
                                     'test': test,
                                     'train_results': train_results,
                                     'test_results': test_results}})

    #    with open('raw_cv_data.pickle', 'wb') as fh:
    #        pickle.dump(output, fh)

        summary_stats = mean_var_calc(output)

        for subject in settings['SUBJECTS']:
            for t in summary_stats[subject]:
                try:
                    global_results[subject][t] += [summary_stats[subject][t]]
                except KeyError:
                    global_results[subject][t] = [summary_stats[subject][t]]
    print(global_results)
    for subject in settings['SUBJECTS']:
        for t in global_results[subject]:
            meanscore = np.mean(global_results[subject][t])
            varscore = np.var(global_results[subject][t])
            print("For {0} mean {1} was "
                  "{2} with sigma {3}".format(subject, t, meanscore, varscore))

    with open('summary_stats.pickle', 'wb') as fh:
        pickle.dump(global_results, fh)


def fit_and_return_parts_and_results(da, metadata, pipe, X, y):
    '''
    function to fit a CV and return a list of
    parts and results
    '''

    train_results = []
    test_results = []
    train_partition = []
    test_partition = []

    cv = utils.Sequence_CV(da.training_segments, metadata)

    for train, test in cv:

        weight = len(y[train]) / sum(y[train])
        weights = [weight if i == 1 else 1 for i in y[train]]
        pipe.fit(X[train], y[train], cls__sample_weight=weights)
        ptest = pipe.predict_proba(X[test])
        ptrain = pipe.predict_proba(X[train])
        # train_score = sklearn.metrics.roc_auc_score(y[train], p_train[:,1])
        # test_score = sklearn.metrics.roc_auc_score(y[test], p_test[:,1])

        # store subject predictions and true labels
        train_results.append(np.hstack([y[train][np.newaxis].T,
                                        ptrain[:, 1][np.newaxis].T]))
        test_results.append(np.hstack([y[test][np.newaxis].T,
                                       ptest[:, 1][np.newaxis].T]))

    return train_partition, test_partition, train_results, test_results


def mean_var_calc(output):

    summary_stats = {}
    global_train = []
    global_test = []
    for subject in output.keys():
        train_results = np.vstack(output[subject]['train_results'])

        trainscore = sklearn.metrics.roc_auc_score(train_results[:, 0],
                                                   train_results[:, 1])
        global_train.append(train_results)
        test_results = np.vstack(output[subject]['test_results'])
        testscore = sklearn.metrics.roc_auc_score(test_results[:, 0],
                                                  test_results[:, 1])
        global_test.append(test_results)
        # mean = np.mean(output[subject]['results'])
        # var = np.var(output[subject]['results'])
        summary_stats.update({subject: {'trainscore': trainscore,
                                        'testscore': testscore}})
    global_train = np.vstack(global_train[:])
    globaltrainscore = sklearn.metrics.roc_auc_score(global_train[:, 0],
                                                     global_train[:, 1])
    global_test = np.vstack(global_test[:])
    globaltestscore = sklearn.metrics.roc_auc_score(global_test[:, 0],
                                                    global_test[:, 1])
    summary_stats['global'] = {'trainscore': globaltrainscore,
                               'testscore': globaltestscore}
    return summary_stats

if __name__ == '__main__':

    main()
