import matplotlib.pyplot as plt
import Constants as const
import pandas as pd
from datetime import datetime
import SyntheticDataGeneration
import DataPreProcessing
import PCLFSMethod as pclfs

# Mainly there would be two objectives and two data applications (synthetic simulation and real world applications)
# of the project
# This main function is related to the simulation part of the objectives

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class SimulationStudy:
    def main(self, analysis, source, id_col, target_col, status, methods, smote_status, n_features, n_redundant, n_repeated, n_classes,
             n_clusters_per_class, n_samples, weight, n_trials, plot_true, csv=False, no_pc=2, step=1):
        csv_name = "Simul1"
        random_state = None

        gen = SyntheticDataGeneration.SyntheticDataGeneration(n_features, n_redundant, n_repeated, n_classes,
                                                              n_clusters_per_class, random_state=None, shuffle=False)
        prep = DataPreProcessing.DataPreProcessing()
        pclfs_obj = pclfs.PCLFSMethod()

        for sample_size in n_samples:
            for trial in range(n_trials):
                for i in range(n_features - (n_redundant + n_repeated)):
                    n_informative = 10
                    print("N_INFORMATIVES: ", n_informative)
                    n_non_informative = n_features - n_informative
                    print(n_informative)

                    # Step1: generate synthetic data
                    result = gen.generate_data(sample_size, n_informative, target_col, weight, source, csv)

                    # Step2: clean data --> Nothing to do with cleaned synthetic data

                    # Step3 & Step4: data splitting & pre-proccesing
                    result = prep.split_and_scale_data(result, id_col, target_col, True, random_state=None)

                    split_pt_data = pd.DataFrame(result['split_data'])

                    split_pt_data.to_csv(const.Constants.SAVE_PATH + source + '\\split_processed_' + csv_name + '.csv',
                                         index=False, header=True)
                    split_ps_data = pd.DataFrame(result['smote_split'])
                    split_ps_data.to_csv(
                        const.Constants.SAVE_PATH + source + '\\smote_split_processed_' + csv_name + '.csv', index=False,
                        header=True)
                    split_ptest_data = pd.DataFrame(result['test_data'])
                    split_ptest_data.to_csv(const.Constants.SAVE_PATH + source + '\\processed_test_' + csv_name + '.csv',
                                            index=False,
                                            header=True)

                    cols = [i for i in split_pt_data.columns if i not in [target_col]]

                    n_features_to_select = None
                    feature_list = [i for i in split_pt_data.columns if i not in [id_col] + [target_col]]
                    n_features = len(feature_list)

                    if analysis == 'pclfs-ext':
                        types = ['basic', 'rfe', 'pclfs']
                        header = True
                        for smote in smote_status:
                            for method in methods:
                                for type in types:
                                    method_fs = method + '_' + type
                                    start_time = datetime.now()

                                    if smote:
                                        result, grid_score, _ = pclfs_obj.new_feature_selection(method_fs, split_ps_data[cols],
                                                                                                split_ptest_data[cols],
                                                                                                split_ps_data[target_col],
                                                                                                split_ptest_data[target_col],
                                                                                                id_col, target_col,
                                                                                                random_state, plot_true, smote,
                                                                                                n_features_to_select, source,
                                                                                                no_pc=2, step=1)
                                    else:
                                        result, grid_score, _ = pclfs_obj.new_feature_selection(method_fs, split_pt_data[cols],
                                                                                                split_ptest_data[cols],
                                                                                                split_pt_data[target_col],
                                                                                                split_ptest_data[target_col],
                                                                                                id_col, target_col,
                                                                                                random_state, plot_true, smote,
                                                                                                n_features_to_select, source,
                                                                                                no_pc=2, step=1)
                                    print(method)
                                    end_time = datetime.now()
                                    duration = end_time - start_time
                                    dataset = "info" + str(n_informative) + "_ss" + str(sample_size)
                                    result.insert(0, 'imbalance_rate', str(weight))
                                    result.insert(1, 'n_informative_given', n_informative)
                                    result.insert(2, 'sample_size', sample_size)
                                    result.insert(3, 'n_total_features', n_features)

                                    print(result['optimal_features'][0][0])
                                    try:
                                        i_occurrence = 0
                                        n_occurrence = 0
                                        for feature_list in result['optimal_features']:
                                            for feature in feature_list:
                                                if feature.startswith('i_'):
                                                    i_occurrence = i_occurrence + 1
                                                elif feature.startswith('n_'):
                                                    n_occurrence = n_occurrence + 1

                                        correct_percentage = ((i_occurrence / n_informative) + ((n_non_informative - n_occurrence) / n_non_informative)) / 2
                                    except ZeroDivisionError:
                                        correct_percentage = ''
                                    print("correct_percentage: %d ", correct_percentage)

                                    result["correct_percentage"] = correct_percentage
                                    csv_name = str(analysis) + str(method) + str(sample_size) + str(n_features)
                                    result.to_csv(const.Constants.SAVE_PATH + source +  "\\" + csv_name + "simulation_results.csv", index=False, mode='a', header=header)
                                    header = False