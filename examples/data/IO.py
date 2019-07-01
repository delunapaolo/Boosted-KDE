import os
import pandas as pd
from scipy.io.matlab import loadmat


def load_data(dataset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(dir_path, dataset_name)
    if not os.path.exists(root_path):
        raise FileNotFoundError('The folder \'%s\' does not exist' % dataset_name)
    files = os.listdir(root_path)

    if dataset_name == 'breast_cancer':
        column_names = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Classification']
        dtype = {'Age': int, 'BMI':float, 'Glucose':int, 'Insulin':float, 'HOMA':float, 'Leptin':float, 'Adiponectin':float, 'Resistin':float, 'MCP.1':float, 'Classification':int}
        separator = ','
        skip_rows = 1
        skip_concatenation = False

    elif dataset_name == 'cardio':
        x = loadmat(os.path.join(root_path, files[0]))
        data = x['X']
        labels = x['y']
        column_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
        DATA = pd.DataFrame(data, columns=column_names)
        DATA['label'] = labels

        skip_concatenation = True

    if not skip_concatenation:
        DATA = pd.DataFrame(columns=column_names)
        for this_file in files:
            DATA = pd.concat((DATA,
                              pd.read_csv(os.path.join(root_path, this_file),
                                          sep=separator,
                                          header=None, skiprows=skip_rows,
                                          names=list(DATA.columns))),
                             ignore_index=True, axis=0)

        # Cast to correct data type
        for var in column_names:
            DATA[var] = DATA[var].map(dtype[var])

    print('Loaded dataset \'%s\', containing %i observations and %i variables:\n\t%s' % (dataset_name, DATA.shape[0], DATA.shape[1], ', '.join(list(DATA.columns))))
    return DATA


def rearrange_results_dict(results):
    # Gather data for plot
    data = pd.DataFrame(columns=['classifier type', 'group', 'performance',
                                 'boosting', 'CV_iteration', 'kernel',
                                 'regularization', 'cost'])
    group_names = list(results.keys())
    for i_group in range(len(group_names)):
        task_names = list(results[group_names[i_group]].keys())
        for i_task in range(len(task_names)):
            data_this_group = results[group_names[i_group]][task_names[i_task]]
            # Extract values that didn't change within CV
            name_parts = task_names[i_task].split('__')
            if len(name_parts) > 2:
                classifier_type = name_parts[0] + '_' + name_parts[2]
            else:
                classifier_type = name_parts[0]
            did_boost = int(name_parts[1]) > 0

            n_CV_iterations = len(data_this_group['CV_performance'])
            for i_iter in range(n_CV_iterations):
                # Get info from this CV iteration
                performance = data_this_group['CV_performance'][i_iter]
                if classifier_type == 'SVM':
                    cost = data_this_group['parameters'][i_iter]['svm__C']
                else:
                    cost = data_this_group['parameters'][i_iter]['svm__nu']
                gamma = data_this_group['parameters'][i_iter]['svm__gamma']
                kernel = data_this_group['parameters'][i_iter]['svm__kernel']

                # Store data
                row = data.shape[0]
                data.at[row, 'classifier type'] = classifier_type
                data.at[row, 'group'] = i_group
                data.at[row, 'performance'] = performance
                data.at[row, 'boosting'] = did_boost
                data.at[row, 'CV_iteration'] = i_iter
                data.at[row, 'kernel'] = kernel
                data.at[row, 'regularization'] = gamma
                data.at[row, 'cost'] = cost
    # Cast columns to correct data types
    data['performance'] = data['performance'].map(float)

    return data
