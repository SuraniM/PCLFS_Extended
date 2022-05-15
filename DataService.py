import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import DataCleaning
import copy
import Constants as const
import DataPreProcessing


class DataService:
    def __init__(self, source, id_col, target_col):

        self.source = source
        self.id_col = id_col

        self.target_col = [target_col]

    def process_data(self, read_csv, data):

        # Data cleaning
        manip = DataCleaning.DataCleaning()
        if data:
            data_org = read_csv
        else:
            data_org = pd.read_csv(const.Constants.READ_PATH + self.source + '/' + read_csv, encoding='latin-1', dtype=str)

        if self.source != "CKD/final":
            result1 = manip.data_cleaning(data_org, self.source, self.target_col)
            data_clean = result1['data']
            churn_data = result1['churn']
            not_churn_data = result1['not_churn']
        else:
            data_clean = data_org

        # categorical columns
        cat_cols = data_clean.nunique()[data_clean.nunique() < 6].keys().tolist()
        cat_cols = [x for x in cat_cols if x not in self.target_col]
        # numerical columns
        num_cols = [x for x in data_clean.columns if x not in cat_cols + self.target_col + [self.id_col]]
        # Binary columns with 2 values
        bin_cols = data_clean.nunique()[data_clean.nunique() == 2].keys().tolist()
        # Columns more than 2 values
        multi_cols = [i for i in cat_cols if i not in bin_cols]

        data = copy.deepcopy(data_clean)  # cleaned original data

        pre = DataPreProcessing.DataPreProcessing()
        result2 = pre.data_pre_processed(data_clean, self.id_col, self.target_col)
        output = {'cleaned_data': result2['cleaned_data'], 'processed_data': result2['processed_data'],
                  'cat_cols': cat_cols, 'num_cols': num_cols, 'bin_cols': bin_cols, 'multi_cols': multi_cols}
        return output



