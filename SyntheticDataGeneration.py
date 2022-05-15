# This gives the gradient value of the

from sklearn import datasets
import pandas as pd
import numpy as np
import Constants as const


class SyntheticDataGeneration():
    def __init__(self, n_features, n_redundant, n_repeated, n_classes, n_clusters_per_class, random_state=None,
                              shuffle=False):
        self.n_features = n_features
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.random_state = random_state
        self.shuffle = shuffle
        pass

    def generate_data(self, sample_size, n_informative, target_col, weights, source, csv):
        read_csv = "data" + str(n_informative) + "_ss" + str(sample_size) + ".csv"

        data = datasets.make_classification(n_samples=sample_size, n_features=self.n_features,
                                            n_informative=n_informative, n_redundant=self.n_redundant,
                                            n_repeated=self.n_repeated, n_classes=self.n_classes,
                                            n_clusters_per_class=self.n_clusters_per_class, weights=weights,
                                            flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                                            shuffle=self.shuffle, random_state=self.random_state)
        predictors_list = data[0]
        df1 = pd.Series(predictors_list.tolist())

        df1 = df1.apply(lambda x: pd.to_numeric(pd.Series(str(x)[1:-1].split(","))))
        n_useful = n_informative + self.n_redundant + self.n_repeated
        column_names = []

        for i in range(self.n_features):
            if i < n_informative:
                col = "i_" + str(i + 1)
                column_names.append(col)
            elif i < n_informative + self.n_redundant:
                col = "re_" + str(i + 1)
                column_names.append(col)
            elif i < n_useful:
                col = "rp_" + str(i + 1)
                column_names.append(col)
            else:
                col = "n_" + str(i + 1)
                column_names.append(col)
        print(column_names)

        df1.columns=column_names

        target_list = data[1]
        df2 = pd.DataFrame()

        df2[target_col] = pd.Series(target_list.tolist())

        df = pd.concat([df1, df2], axis=1, sort=False)

        ID_col = list(range(1, sample_size+1, 1))
        df.insert(0, 'ID_col', ID_col)
        if csv:
            df.to_csv(const.Constants.SAVE_PATH + source + '/' + read_csv, index=False, header=True)

        return df