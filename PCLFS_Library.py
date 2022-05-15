import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
import Grid_Score_Calc
import Constants as const
import plotly.express as px
import time
from imblearn.over_sampling import SMOTE
import Grid_Score_Calc as gs_calc
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import DefineClassifiers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
import DefineClassifiers


class PCLFS_Library:

    def pclfs_main_code(self, method, classifier, training_x, testing_x, training_y, testing_y, id_col, target_col,
                              random_state, plot_true, smote, n_features_to_select, no_pc, step):
        train = pd.DataFrame(training_y, columns=[target_col])
        total_df = pd.concat([training_x, train])

        start_time = int(round(time.time() * 1000))

        pc_cols = []
        for i in range(no_pc):
            pc_name = "PC" + str(i + 1)
            pc_cols.append(pc_name)
        pca = decomposition.PCA(n_components=no_pc)

        principal_components = pca.fit_transform(training_x)
        pca_data = pd.DataFrame(principal_components, columns=pc_cols)

        pca_data[target_col] = training_y.values
        pca_data[target_col] = pca_data[target_col].replace({1: "abnormal", 0: "normal"})

        loadings = pd.DataFrame(pca.components_.T, columns=pc_cols, index=training_x.columns.values)

        pc_abs_sum = abs(loadings[pc_cols[0]])
        for i in range(no_pc - 1):
            pc_abs_sum = pc_abs_sum + abs(loadings[pc_cols[i + 1]])
        loadings['PC_abs_sum'] = pc_abs_sum
        return loadings

    def pclfs_validate_main_code(self, method, classifier, training_x, testing_x, training_y, testing_y, id_col, target_col,
                              random_state, plot_true, smote, n_features_to_select, no_pc, step):
        tes = pd.DataFrame(testing_y, columns=[target_col])
        total_testing_df = pd.concat([testing_x, tes])
        classifier_obj = DefineClassifiers.DefineClassifiers()

        start_time = int(round(time.time() * 1000))

        pc_cols = []
        for i in range(no_pc):
            pc_name = "PC" + str(i + 1)
            pc_cols.append(pc_name)

        loadings = self.identify_pc_loadings(no_pc, training_x, training_y, pc_cols, target_col)
        pca_df = loadings.sort_values(by=['PC_abs_sum'], ascending=False)
        # print(pca_df)
        feature_importance = list(pca_df.index)
        print('important: ')
        print(feature_importance)

        grid_scores = []
        precision_scores = []
        recall_scores = []

        f_set = []
        T = np.zeros(len(feature_importance))
        skf = StratifiedKFold(n_splits=5, random_state=None)
        training_x = training_x.reset_index(drop=True)
        training_y = training_y.reset_index(drop=True)

        # X is the feature set and y is the target
        for train_index, val_index in skf.split(training_x, training_y):
            X_train, X_validate = training_x.loc[train_index], training_x.loc[val_index]
            y_train, y_validate = training_y.loc[train_index], training_y.loc[val_index]

            loadings_t = self.identify_pc_loadings(no_pc, X_train, y_train, pc_cols, target_col)
            pca_df_t = loadings_t.sort_values(by=['PC_abs_sum'], ascending=False)

            feature_importance_t = list(pca_df_t.index)
            print(feature_importance_t)

            # for i in feature_importance:
            for i in range(0, len(feature_importance_t), step):
                for j in range(step):
                    try:
                        f_set.append(feature_importance_t[i + j])
                    except (IndexError):
                        pass

                train_x = pd.DataFrame(X_train[f_set])
                classifier.fit(train_x, y_train)
                pca_predictions = classifier.predict(X_validate[f_set])
                pc_f1score = f1_score(y_validate, pca_predictions)
                T[i] = T[i] + pc_f1score

                pc_recallscore = recall_score(y_validate, pca_predictions)
                pc_precision = precision_score(y_validate, pca_predictions)
                recall_scores.append(pc_recallscore)
                precision_scores.append(pc_precision)
        grid_scores = (T / 5).tolist()
        max_f1score = max(grid_scores)
        selected_n_features = grid_scores.index(max_f1score) + 1
        optimal_features = list(feature_importance[0:selected_n_features])
        print(optimal_features)

        classifier_new, cf, best_parameters, cf = classifier_obj.get_classifier(method, training_x[optimal_features],
                                                                                training_y,
                                                                                n_features_to_select, rfe=False)
        classifier_new.fit(training_x[optimal_features], training_y)
        org_pred = classifier_new.predict(testing_x[optimal_features])
        final_f1score = f1_score(testing_y, org_pred)
        final_recall = recall_score(testing_y, org_pred)
        final_mcc = matthews_corrcoef(testing_y, org_pred)
        final_precision = precision_score(testing_y, org_pred)

        duration = int(round(time.time() * 1000)) - start_time

        output = {'fs_method': method, 'max_f1score': max_f1score, 'final_f1score': final_f1score,
                  'final_precision': final_precision, 'final_recall': final_recall, 'final_mcc': final_mcc,
                  'duration': duration, 'smote': smote,
                  'selected_n_features': selected_n_features, 'optimal_features': optimal_features,
                  'grid_scores': grid_scores, 'loadings': loadings}

        return output

    def pclfs_cv_main_code(self, method, classifier, training_x, testing_x, training_y, testing_y, id_col, target_col,
                              random_state, plot_true, smote, n_features_to_select, no_pc, step):
        tes = pd.DataFrame(testing_y, columns=[target_col])
        total_testing_df = pd.concat([testing_x, tes])
        classifier_obj = DefineClassifiers.DefineClassifiers()

        start_time = int(round(time.time() * 1000))

        pc_cols=[]
        for i in range(no_pc):
            pc_name = "PC"+str(i+1)
            pc_cols.append(pc_name)

        loadings = self.identify_pc_loadings(no_pc, training_x, training_y, pc_cols, target_col)
        pca_df = loadings.sort_values(by=['PC_abs_sum'], ascending=False)
        # print(pca_df)
        feature_importance = list(pca_df.index)
        print('important: ')
        print(feature_importance)

        # f_set = [feature_importance[0]]
        grid_scores = []
        precision_scores = []
        recall_scores =[]

        f_set = []
        T = np.zeros(len(feature_importance))
        skf = StratifiedKFold(n_splits=5, random_state=None)
        training_x = training_x.reset_index(drop=True)
        training_y = training_y.reset_index(drop=True)

        # X is the feature set and y is the target
        for train_index, val_index in skf.split(training_x, training_y):
            X_train, X_validate = training_x.loc[train_index], training_x.loc[val_index]
            y_train, y_validate = training_y.loc[train_index], training_y.loc[val_index]

            loadings_t = self.identify_pc_loadings(no_pc, X_train, y_train, pc_cols, target_col)
            pca_df_t = loadings_t.sort_values(by=['PC_abs_sum'], ascending=False)
            # print(pca_df)
            feature_importance_t = list(pca_df_t.index)
            print(feature_importance_t)

            # for i in feature_importance:
            for i in range(0, len(feature_importance_t), step):
                for j in range(step):
                    try:
                        f_set.append(feature_importance_t[i+j])
                    except (IndexError):
                        pass

                train_x = pd.DataFrame(X_train[f_set])
                classifier.fit(train_x, y_train)
                pca_predictions = classifier.predict(X_validate[f_set])
                pc_f1score = f1_score(y_validate, pca_predictions)
                T[i] = T[i] + pc_f1score

                pc_recallscore = recall_score(y_validate, pca_predictions)
                pc_precision = precision_score(y_validate, pca_predictions)
                recall_scores.append(pc_recallscore)
                precision_scores.append(pc_precision)
        # print(T)
        grid_scores = (T/5).tolist()
        max_f1score = max(grid_scores)
        selected_n_features = grid_scores.index(max_f1score) + 1
        optimal_features = list(feature_importance[0:selected_n_features])
        print(optimal_features)

        classifier_new, cf, best_parameters, cf = classifier_obj.get_classifier(method, training_x[optimal_features], training_y,
                                                                            n_features_to_select, rfe=False)
        classifier_new.fit(training_x[optimal_features], training_y)
        org_pred = classifier_new.predict(testing_x[optimal_features])
        final_f1score = f1_score(testing_y, org_pred)
        final_recall = recall_score(testing_y, org_pred)
        final_mcc = matthews_corrcoef(testing_y, org_pred)
        final_precision = precision_score(testing_y, org_pred)

        duration = int(round(time.time() * 1000)) - start_time

        output = {'fs_method': method, 'max_f1score': max_f1score, 'final_f1score': final_f1score,
                               'final_precision': final_precision, 'final_recall': final_recall, 'final_mcc': final_mcc,
                               'duration': duration, 'smote': smote,
                               'selected_n_features': selected_n_features, 'optimal_features': optimal_features,
                  'grid_scores': grid_scores, 'loadings': loadings}

        return output

    def identify_pc_loadings(self, no_pc, training_x, training_y, pc_cols, target_col):

        # print(cols)
        pca = decomposition.PCA(n_components=no_pc)

        principal_components = pca.fit_transform(training_x)
        pca_data = pd.DataFrame(principal_components, columns=pc_cols)

        pca_data[target_col] = training_y.values
        pca_data[target_col] = pca_data[target_col].replace({1: "abnormal", 0: "normal"})

        loadings = pd.DataFrame(pca.components_.T, columns=pc_cols, index=training_x.columns.values)

        pc_abs_sum = abs(loadings[pc_cols[0]])
        for i in range(no_pc - 1):
            pc_abs_sum = pc_abs_sum + abs(loadings[pc_cols[i + 1]])
        loadings['PC_abs_sum'] = pc_abs_sum
        return loadings

    def identify_cv_pc_loadings(self, no_pc, training_x, training_y, pc_cols, target_col):
        cv = [1, 2]
        loadings = pd.DataFrame()
        X_train, X_test, y_train, y_test = train_test_split(training_x, training_y, test_size=0.5, random_state=42)
        for cv in cv:
            pca = decomposition.PCA(n_components=no_pc)
            if cv == 1:
                principal_components = pca.fit_transform(X_train)
            else:
                principal_components = pca.fit_transform(X_test)
            pca_data = pd.DataFrame(principal_components, columns=pc_cols)

            if cv == 1:
                pca_data[target_col] = y_train.values
            else:
                pca_data[target_col] = y_test.values
            pca_data[target_col] = pca_data[target_col].replace({1: "abnormal", 0: "normal"})

            loadings_tmp = pd.DataFrame(pca.components_.T, columns=pc_cols, index=X_train.columns.values)

            pc_abs_sum = abs(loadings_tmp[pc_cols[0]])
            for i in range(no_pc - 1):
                pc_abs_sum = pc_abs_sum + abs(loadings_tmp[pc_cols[i + 1]])

            if cv == 1:
                loadings['PC_abs_sum1'] = pc_abs_sum
            else:
                loadings['PC_abs_sum2'] = pc_abs_sum
        loadings['PC_abs_sum'] = loadings['PC_abs_sum1'] + loadings['PC_abs_sum2']
        return loadings