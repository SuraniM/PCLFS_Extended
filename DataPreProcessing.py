from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy
import Constants as const
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import scale

class DataPreProcessing:

    def __init__(self):
        pass

    def scale_data(self, org_data, source, id_col, target_col):
        cols = [i for i in org_data.columns if i not in id_col + target_col]
        keep = org_data[['OVERALL_DIAGNOSIS', 'id']]
        pre_processed_data = pd.DataFrame(scale(org_data[cols]))
        pre_processed_data = pd.concat([pre_processed_data, keep])
        return pre_processed_data

    def split_and_scale_data(self, data, id_col, target_col, prep, random_state):

        data = data.replace('?', 0)

        cols = [i for i in data.columns if i not in id_col + target_col]

        if prep:
            result = self.data_pre_processed(data, target_col, id_col)
            data = result['processed_data']

        else:
            data = data

        cols = [i for i in data.columns if i not in id_col + target_col]

        # Splitting train and test data
        train, test = train_test_split(data, test_size=.3, random_state=random_state)
        test_x = test[cols]
        test_y = test[target_col]
        train_x = train[cols]
        train_y = train[target_col]

        # oversampling minority class using smote
        os = SMOTE(random_state=0)
        os_train_x, os_train_y = os.fit_sample(train_x, train_y)
        os_train_x = pd.DataFrame(data=os_train_x, columns=cols)
        os_train_y = pd.DataFrame(data=os_train_y, columns=[target_col])

        df_t = pd.concat([train_x, train_y], axis=1, sort=False)
        df_test = pd.concat([test_x, test_y], axis=1, sort=False)
        df_s = pd.concat([os_train_x, os_train_y], axis=1, sort=False)

        output = {'split_data': df_t, 'test_data': df_test, 'smote_split': df_s}

        return output

    def data_pre_processed(self, data, id_col, target_col):

        # categorical columns
        cat_cols = data.nunique()[data.nunique() < 6].keys().tolist()
        cat_cols = [x for x in cat_cols if x not in [target_col]]
        # numerical columns
        num_cols = [x for x in data.columns if x not in cat_cols + [target_col] + [id_col]]
        # Binary columns with 2 values
        bin_cols = data.nunique()[data.nunique() == 2].keys().tolist()
        # Columns more than 2 values
        multi_cols = [i for i in cat_cols if i not in bin_cols]

        # Label encoding Binary columns
        le = LabelEncoder()
        for i in bin_cols:
            data[i] = le.fit_transform(data[i])

        std = StandardScaler()
        num_columns = pd.DataFrame()
        for i in num_cols:
            num_columns[i] = pd.to_numeric(data[i], errors='coerce')
        # num_df = num_df.astype(np.float64)
        print(num_columns.dtypes)
        scaled = std.fit_transform(num_columns[num_cols])
        scaled = pd.DataFrame(scaled, columns=num_cols)

        print(scaled.shape)

        cleaned_data = copy.deepcopy(data)

        # dropping original values merging scaled values for numerical columns
        scaled_data = data.drop(columns=num_cols, axis=1)
        scaled_data = scaled_data.merge(scaled, left_index=True, right_index=True, how="left")

        scaled_data = scaled_data.dropna()

        output = {'cleaned_data': cleaned_data, 'processed_data': scaled_data}
        return output
