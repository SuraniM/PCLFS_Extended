from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn_rvm import EMRVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


class DefineClassifiers():

    def get_classifier(self, method, training_x, training_y, n_features_to_select, rfe):
        random_state = 100
        # random_state = None
        if 'logit' in method:
            classifier = LogisticRegression(solver='liblinear', random_state=random_state)
            grid = {}
            cf = 'coefficients'

        elif 'lgbm_c' in method:
            classifier = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                        learning_rate=0.5,
                                        max_depth=7, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                        n_estimators=100, n_jobs=-1, num_leaves=500, objective='binary',
                                        random_state=random_state,
                                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                                        subsample_for_bin=200000,
                                        subsample_freq=0)
            grid = {}
            cf = 'features'

        elif 'xgb' in method:
            classifier = XGBClassifier(objective='binary:logistic')
            grid = {
                'n_estimators': [100, 110, 150],
                'max_depth': [4, 5, 7, 10],
                'learning_rate': [0.0001, 0.001, 0.01]
            }
            cf = 'features'

        elif 'decision_tree' in method:
            classifier = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=random_state)
            grid = {
                'max_depth': [4, 5, 7, 10, 11]
            }
            cf = 'features'

        elif 'knn' in method:
            classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                                              n_jobs=1, n_neighbors=5, p=2, weights='uniform')
            grid = {}
            cf = ''

        elif 'rfc' in method:
            classifier = RandomForestClassifier(criterion="gini", n_estimators=100, random_state=random_state)
            grid = {
                'max_depth': [10, 20, 30, 40],
                'max_features': ['log2', 'sqrt']
            }
            cf = 'features'

        elif 'gnb' in method:
            classifier = GaussianNB(priors=None)
            grid = {}
            cf = 'features'

        elif 'svm_lin' in method:
            classifier = SVC(kernel='linear', C=1)
            grid = {
                # 'C': [10 ** i for i in range(-10, 10, 1)]
            }
            cf = 'coefficients'

        elif 'svm_rbf' in method:
            classifier = SVC(kernel='rbf')
            grid = {
                'C': [10 ** i for i in range(-10, 10, 2)],
                'gamma': [2 ** i for i in range(-10, 10, 1)]
            }
            cf = ''

        elif 'rvm_lin' in method:
            classifier = EMRVC(kernel='linear')
            grid = {}
            cf = ''

        elif 'rvm_rbf' in method:
            classifier = EMRVC(kernel='rbf')
            grid = {}
            cf = ''

        else:
            classifier = ''
            grid = {}
            cf = ''

        if grid == {}:
            best_parameters = {}
        else:
            gd_sr = GridSearchCV(estimator=classifier,
                                 param_grid=grid,
                                 scoring='f1',  # note the use of scoring here.
                                 cv=5, iid=False,
                                 n_jobs=-1)

            gd_sr.fit(training_x, training_y)
            best_parameters = gd_sr.best_params_
            classifier = gd_sr.best_estimator_

        if rfe:
            if n_features_to_select is None and 'ext' not in method:
                classifier = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(5), scoring='f1')
            else:
                classifier = RFE(estimator=classifier, n_features_to_select=n_features_to_select)

        return classifier, cf, best_parameters, cf