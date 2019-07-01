# Main functions
from boosted_KDE import KDEBoosting

# Third-party numerical libraries
import numpy as np
from scipy.stats._continuous_distns import uniform_gen
from sklearn.pipeline import Pipeline, clone
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as compute_average_precision_score


def compute_performance(classification_tasks, X, y,
                        n_splits=3, n_inner_splits_tuning=3,
                        fraction_tuning=.25,
                        n_jobs=None,
                        random_seed=None, verbose=False):

    # Initialize variables
    base_SVM = None
    base_ocSVM = None
    clf = None
    SVM_params = None
    ocSVM_params = None
    params = None

    # Unpack tasks
    classifier_types = [i[0] for i in classification_tasks]
    boosting_iterations = [i[1] for i in classification_tasks]
    boosting_iterations_set = set(boosting_iterations)
    user_parameters = [i[2] for i in classification_tasks]
    n_tasks = len(classification_tasks)

    # Initialize output variable
    results = dict()
    results_keys = list()
    for i_task in range(n_tasks):
        this_classifier_type = classifier_types[i_task]
        this_boosting = boosting_iterations[i_task]
        this_user_params = user_parameters[i_task]
        if this_user_params is None:
            key_name = '%s__%i' % (this_classifier_type, this_boosting)
        else:
            key_name = '%s__%i__%s' % (this_classifier_type, this_boosting, '_'.join(list(this_user_params.keys())))
        # Store key name
        results_keys.append(key_name)
        # Initialize dictionary's subfield
        results[key_name] = dict(classifier_type=this_classifier_type,
                                 parameters=list(),
                                 CV_performance=np.zeros((n_splits, ), dtype=float),
                                 boosting_iterations=this_boosting)

    # Create base classifiers
    if 'SVM' in classifier_types:
        base_SVM = Pipeline([('scaler', StandardScaler(with_mean=True, with_std=True, copy=True)),
                             ('svm', SVC(random_state=random_seed))])
    if 'ocSVM' in classifier_types:
        base_ocSVM = Pipeline([('scaler', StandardScaler(with_mean=True, with_std=True, copy=True)),
                             ('svm', OneClassSVM())])

    # Create generators of CV splits
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=random_seed)
    # Initialize counter for outer CV loop
    i_iter = 0

    for train_idx, test_idx in outer_cv.split(X, y):
        if verbose:
            print('CV iteration %i/%i' % (i_iter + 1, n_splits))

        # Select training and test set
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        # Tune hyperparameters
        if fraction_tuning > 0:
            # Hold out a validation set for hyperparameter tuning
            X_train, X_tuning, y_train, y_tuning = train_test_split(
                    X_train, y_train, test_size=fraction_tuning, stratify=y_train,
                    random_state=random_seed)

            # Compute sample weights of train and test set
            weights_tuning = dict()
            for k in boosting_iterations_set:
                weights_tuning['k_%i' % k] = KDEBoosting(X_tuning, k_iterations=k,
                                                         n_jobs=n_jobs).normalized_weights

            # Define distribution of parameters
            if 'SVM' in classifier_types:
                SVM_params = dict(svm__kernel=['linear', 'rbf'],
                                  svm__gamma=uniform_gen(a=.01, b=10),
                                  svm__C=uniform_gen(a=.01, b=10))
            if 'ocSVM' in classifier_types:
                # Define distribution of parameters
                ocSVM_params = dict(svm__kernel=['linear', 'rbf'],
                                    svm__gamma=uniform_gen(a=.01, b=10),
                                    svm__nu=uniform_gen(a=0, b=1))

            # Search hyperparameters
            for i_task in range(n_tasks):
                this_classifier_type = classifier_types[i_task]
                this_boosting = boosting_iterations[i_task]
                this_user_params = user_parameters[i_task]

                # Initialize classifier, and take parameters to optimize
                if this_classifier_type == 'SVM':
                    clf = clone(base_SVM)
                    params = SVM_params.copy()
                elif this_classifier_type == 'ocSVM':
                    clf = clone(base_ocSVM)
                    params = ocSVM_params.copy()
                # Enforce parameters provided by user
                if this_user_params is not None:
                    params_dict = {'svm__%s' % key: val for key, val in this_user_params.items()}
                    clf.set_params(**params_dict)
                    # Remove user=provided parameters from list of parameters to tune
                    [params.pop(i, None) for i in list(params_dict.keys())]

                # Get sample weights
                w = weights_tuning['k_%i' % this_boosting]

                # Create a CV object
                inner_cv_hp_tuning = StratifiedKFold(n_splits=n_inner_splits_tuning,
                                                     shuffle=False, random_state=random_seed)
                # Run search
                this_search = RandomizedSearchCV(clf, params, scoring='average_precision',
                                        error_score=np.nan, iid=True, refit=False,
                                        cv=inner_cv_hp_tuning, n_iter=50,
                                        n_jobs=n_jobs, random_state=random_seed)
                this_search.fit(X_tuning, y_tuning, svm__sample_weight=w)
                # Get best parameters
                optimized_parameters = this_search.best_params_
                # Append user-provided parameters
                if this_user_params is not None:
                    user_params = list(this_user_params.keys())
                    for param in user_params:
                        optimized_parameters['svm__%s' % param] = this_user_params[param]
                # Store parameters
                results[results_keys[i_task]]['parameters'].append(optimized_parameters)

        # Compute sample weights of train and test set
        weights_train = dict()
        weights_test = dict()
        for k in boosting_iterations_set:
            weights_train['k_%i' % k] = KDEBoosting(X_train, k_iterations=k,
                                                    n_jobs=n_jobs).normalized_weights
            weights_test['k_%i' % k] = KDEBoosting(X_test, k_iterations=k,
                                                   n_jobs=n_jobs).normalized_weights

        # Train classifiers
        for i_task in range(n_tasks):
            this_classifier_type = classifier_types[i_task]
            this_boosting = boosting_iterations[i_task]
            # Initialize classifier
            if this_classifier_type == 'SVM':
                clf = clone(base_SVM)
            elif this_classifier_type == 'ocSVM':
                clf = clone(base_ocSVM)
            # Assign tuned parameters
            params = results[results_keys[i_task]]['parameters'][i_iter]
            clf.set_params(**params)
            # Get weights
            w_train = weights_train['k_%i' % this_boosting]
            w_test = weights_test['k_%i' % this_boosting]

            # Train classifier
            clf.fit(X_train, y_train, svm__sample_weight=w_train)
            # Compute performance
            y_pred = clf.predict(X_test)
            results[results_keys[i_task]]['CV_performance'][i_iter] = compute_average_precision_score(
                        y_test, y_pred, pos_label=0, sample_weight=w_test)

        # Log progress and outcome
        if verbose:
            for i_task in range(n_tasks):
                this_classifier_type = classifier_types[i_task]
                this_boosting = boosting_iterations[i_task]
                this_user_params = user_parameters[i_task]
                # Assemble message to print
                if this_boosting > 0:
                    prefix = 'Weighted (k=%i) ' % this_boosting
                else:
                    prefix = ''
                if this_user_params is None:
                    suffix = ''
                else:
                    suffix = ' (forced: ' + ', '.join(['%s=%s' % (key, val) for key, val in this_user_params.items()]) + ')'
                msg = '%s%s%s: ' % (prefix, this_classifier_type, suffix)
                # Add info on parameters
                msg += ', '.join(['%s=%s' % (key.replace('svm__', ''), val if isinstance(val, str) else '%.4f' % val) for key, val in results[results_keys[i_task]]['parameters'][i_iter].items()])
                # Add info on performance
                msg += ', performance: %.5f' % results[results_keys[i_task]]['CV_performance'][i_iter]
                # Print message
                print('\t%s' % msg)

        # Increase counter of outer CV loop
        i_iter += 1

    # Print summary of mean performance
    if verbose:
        print('Mean CV performance')
        for i_task in range(n_tasks):
            this_classifier_type = classifier_types[i_task]
            this_boosting = boosting_iterations[i_task]
            this_user_params = user_parameters[i_task]
            performance = results[results_keys[i_task]]['CV_performance']
            # Assemble message to print
            if this_boosting > 0:
                prefix = 'Weighted (k=%i) ' % this_boosting
            else:
                prefix = ''
            if this_user_params is None:
                suffix = ''
            else:
                suffix = ' (forced: ' + ', '.join(['%s=%s' % (key, val) for key, val in this_user_params.items()]) + ')'
            msg = '%s%s%s: %.5f' % (prefix, this_classifier_type, suffix, performance.mean())
            # Print message
            print('\t%s' % msg)
        print('\n')

    return results
