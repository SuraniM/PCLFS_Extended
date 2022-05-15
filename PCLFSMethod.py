import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.metrics import confusion_matrix
import Grid_Score_Calc as gs_calc
import DefineClassifiers
from sklearn.metrics import precision_score, recall_score
import PCLFS_Library
from sklearn.metrics import accuracy_score


class PCLFSMethod:
    def __init__(self):
        pass

    def new_feature_selection(self, method, training_x, testing_x, training_y, testing_y, id_col, target_col,
                              random_state, plot_true, smote, n_features_to_select, source, no_pc, step):
        extended = True

        classifier_obj = DefineClassifiers.DefineClassifiers()
        classifier, cf, best_parameters, cf = classifier_obj.get_classifier(method, training_x, training_y,
                                                                        n_features_to_select, rfe=False)
        features = [i for i in training_x.columns if i not in id_col + target_col]
        n_features = len(features)

        if 'pclfs' in method:
            pclfs_obj = PCLFS_Library.PCLFS_Library()
            start_time = int(round(time.time() * 1000))

            loadings = pclfs_obj.pclfs_main_code(method, classifier, training_x, testing_x, training_y, testing_y, id_col, target_col,
                              random_state, plot_true, smote, n_features_to_select, no_pc, step)
            pca_df = loadings.sort_values(by=['PC_abs_sum'], ascending=False)
            feature_importance = list(pca_df.index)
            grid_scores = []

            precision_scores = []
            recall_scores = []
            accuracy_scores = []
            f_set = []
            # for i in feature_importance:
            for i in range(0, len(feature_importance), step):
                for j in range(step):
                    try:
                        f_set.append(feature_importance[i + j])
                    except (IndexError):
                        pass
                print(f_set)
                train_x = pd.DataFrame(training_x[f_set])
                classifier.fit(train_x, training_y)
                pca_predictions = classifier.predict(testing_x[f_set])
                pc_f1score = f1_score(testing_y, pca_predictions)
                grid_scores.append(pc_f1score)
                print(confusion_matrix(testing_y, pca_predictions).ravel())
                pc_recallscore = recall_score(testing_y, pca_predictions)
                pc_precision = precision_score(testing_y, pca_predictions)
                pc_accuracy = accuracy_score(testing_y, pca_predictions)
                recall_scores.append(pc_recallscore)
                precision_scores.append(pc_precision)
                accuracy_scores.append(pc_accuracy)

            # print(len(grid_scores))
            max_f1score = max(grid_scores)
            final_f1score = max_f1score
            final_precision = max(precision_scores)
            final_accuracy = max(accuracy_scores)
            final_recall = max(recall_scores)
            # selected_n_features = step*(grid_scores.index(max_f1score) + 1)
            selected_n_features = grid_scores.index(max_f1score) + 1
            optimal_features = list(feature_importance[0:selected_n_features])
            duration = int(round(time.time() * 1000)) - start_time

        elif 'rfe' in method:

            start_time = int(round(time.time() * 1000))
            classifier, cf, best_parameters, cf = classifier_obj.get_classifier(method, training_x, training_y,
                                                                                n_features_to_select, rfe=False)
            mp = classifier.fit(training_x, training_y)
            pred = classifier.predict(testing_x)
            f1 = f1_score(testing_y, pred)
            print(f1)

            rfe_classifier = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='f1')
            rfe_classifier.fit(training_x, training_y)
            grid_scores = list(rfe_classifier.grid_scores_)
            ranks = rfe_classifier.ranking_

            df = pd.DataFrame(training_x.columns, columns=['cols'])

            rfe_predictions = rfe_classifier.predict(testing_x)
            final_f1score = f1_score(testing_y, rfe_predictions)
            final_precision = precision_score(testing_y, rfe_predictions)
            final_accuracy = accuracy_score(testing_y, rfe_predictions)
            final_recall = recall_score(testing_y, rfe_predictions)
            max_f1score = max(grid_scores)

            selected_n_features = rfe_classifier.n_features_
            df['ranks'] = rfe_classifier.ranking_
            df = df.sort_values(by=['ranks'], ascending=True)
            print(df)
            if cf == "coefficients":
                model_coef = rfe_classifier.estimator_.coef_[0]
            elif cf == "features":
                model_coef = rfe_classifier.estimator_.feature_importances_
            else:
                model_coef = []
            df = df[df['ranks']==1]
            df['model_coef'] = np.absolute(model_coef)
            df = df.sort_values(by='model_coef', ascending=False)
            optimal_features = list(df['cols'])[0:selected_n_features]

            duration = int(round(time.time() * 1000)) - start_time
            loadings = {}

        elif 'basic' in method:
            start_time = int(round(time.time() * 1000))

            classifier.fit(training_x, training_y)
            predictions = classifier.predict(testing_x)
            max_f1score = f1_score(testing_y, predictions)
            final_f1score = max_f1score
            final_precision = precision_score(testing_y, predictions)
            final_accuracy = accuracy_score(testing_y, predictions)
            final_recall = recall_score(testing_y, predictions)
            optimal_features = training_x.columns.values
            selected_n_features = len(optimal_features)
            grid_scores = []
            duration = int(round(time.time() * 1000)) - start_time
            loadings = {}

        else:
            optimal_features = []
            selected_n_features = 0
            grid_scores = []
            f1score_smote = 0
            max_f1score = 0
            final_f1score = max_f1score
            duration = 0
            final_precision = 0
            final_accuracy = 0
            final_recall = 0
            loadings = {}

        result = pd.DataFrame({'fs_method': [method], 'max_f1score': [max_f1score], 'final_f1score': [final_f1score],
                               'final_precision': [final_precision], 'final_accuracy': [final_accuracy],
                               'final_recall': [final_recall],
                               'smote': [smote], 'selected_n_features': [selected_n_features],
                               'optimal_features': [optimal_features]})
        print(method, selected_n_features, final_f1score)
        print(grid_scores)

        new_features = []
        if extended and 'pclfs' in method:

            grids_results = gs_calc.get_best_grid_score(grid_scores, selected_n_features, step, method, plot_true, smote, source)

            method = method + '-ext'
            new_n_features = grids_results['n_select_features']
            new_features = list(optimal_features[0:new_n_features])
            new_max_f1score = grids_results['select_f1_score']

            new_train_data = training_x[new_features]
            new_test_data = testing_x[new_features]

            new_features = list(optimal_features[0:new_n_features])
            classifier, cf, best_parameters, cf = classifier_obj.get_classifier(method, training_x, training_y,
                                                                                    n_features_to_select, rfe=False)

            classifier.fit(new_train_data, training_y)
            new_predictions = classifier.predict(new_test_data)
            new_final_f1score = f1_score(testing_y, new_predictions)
            new_final_precision = precision_score(testing_y, new_predictions)
            new_final_accuracy = accuracy_score(testing_y, new_predictions)
            new_final_recall = recall_score(testing_y, new_predictions)
            new_optimal_features = training_x.columns.values
            print("extended", new_n_features, new_final_f1score)

            new_result = pd.DataFrame(
                {'fs_method': [method], 'max_f1score': [new_max_f1score], 'final_f1score': [new_final_f1score],
                 'final_precision': [new_final_precision],'final_accuracy': [new_final_accuracy],
                 'final_recall': [new_final_recall],
                 'smote': [smote],
                 'selected_n_features': [new_n_features], 'optimal_features': [new_features]})
            result = result.append(new_result)

        return result, grid_scores, loadings
