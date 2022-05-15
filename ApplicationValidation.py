import Constants as const
import pandas as pd
from datetime import datetime
import PCLFSMethod as pclfs
import DataPreProcessing
import DataCleaning


class ApplicationValidation:

    def main(self, analysis, source, csv_name, id_col, target_col, status, methods, smote, no_pc, weights,
             n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1):
        smote_status = smote

        random_state = None  #111

        pclfs_obj = pclfs.PCLFSMethod()
        prep = DataPreProcessing.DataPreProcessing()
        cleaning = DataCleaning.DataCleaning()

        for trail in range(trial):

            if status.lower() == 'new':

                # Step1: read original data
                org_data = pd.read_csv(const.Constants.READ_PATH + source + '/' + csv_name + '.csv', encoding='latin-1') # , dtype=str

                # Step2: clean data
                cleaned_data = cleaning.data_cleaning(org_data, source, target_col)
                cleaned_data['data'].to_csv(const.Constants.SAVE_PATH + source + '\\cleaned90_' + csv_name, index=False, header=True)

                # Step3 & Step4: data splitting & pre-proccesing
                result = prep.split_and_scale_data(cleaned_data['data'], id_col, target_col, False, random_state=None)

                split_pt_data = pd.DataFrame(result['split_data'])
                split_pt_data.to_csv(const.Constants.SAVE_PATH + source + '\\split_processed_' + csv_name+ '.csv', index=False, header=True)
                split_ps_data = pd.DataFrame(result['smote_split'])
                split_ps_data.to_csv(const.Constants.SAVE_PATH + source + '\\smote_split_processed_' + csv_name+ '.csv', index=False,
                                     header=True)
                split_ptest_data = pd.DataFrame(result['test_data'])
                split_ptest_data.to_csv(const.Constants.SAVE_PATH + source + '\\processed_test_' + csv_name + '.csv', index=False,
                                        header=True)

                status = 'old'

            if status.lower() == 'old':
                if no_of_sub_sets is None:
                    split_pt_data = pd.read_csv(const.Constants.READ_PATH + source + '/split_processed_' + csv_name + '.csv', encoding='latin-1')
                    split_ps_data = pd.read_csv(const.Constants.READ_PATH + source + '/smote_split_processed_' + csv_name + '.csv', encoding='latin-1')
                    split_ptest_data = pd.read_csv(const.Constants.READ_PATH + source + '/processed_test_' + csv_name + '.csv', encoding='latin-1')

                    cols = [i for i in split_pt_data.columns if i not in [target_col]]
                    print(split_pt_data)

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
                                    if smote:
                                        result, grid_score, _ = pclfs_obj.new_feature_selection(method_fs,
                                                                                                split_ps_data[cols],
                                                                                                split_ptest_data[cols],
                                                                                                split_ps_data[target_col],
                                                                                                split_ptest_data[target_col],
                                                                                                id_col, target_col,
                                                                                                random_state,
                                                                                                plot_true, smote,
                                                                                                n_features_to_select, source,
                                                                                                no_pc, step=1)
                                    else:
                                        result, grid_score, _ = pclfs_obj.new_feature_selection(method_fs,
                                                                                                split_pt_data[cols],
                                                                                                split_ptest_data[cols],
                                                                                                split_pt_data[target_col],
                                                                                                split_ptest_data[target_col],
                                                                                                id_col, target_col,
                                                                                                random_state,
                                                                                                plot_true, smote,
                                                                                                n_features_to_select, source,
                                                                                                no_pc, step=1)

                                    result.to_csv(const.Constants.SAVE_PATH + source + '\\accuracy_measures_Application' +
                                              ".csv", index=False, mode='a', header=header)
                                    header = False
                status = 'new'