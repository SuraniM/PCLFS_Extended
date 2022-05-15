import numpy as np

class DataCleaning:

    def __init__(self):
        pass

    def data_cleaning(self, data, source, target_col):
        if source == 'telcom' or source == 'RQ2_ext_rfe':

            # Replacing spaces with null values in total charges column
            data['TotalCharges'] = data["TotalCharges"].replace(" ", np.nan)

            # Dropping null values from total charges column which contain .15% missing data
            data = data[data["TotalCharges"].notnull()]
            print(len(data))

            data = data.reset_index()[data.columns]

            # convert to float type
            data["TotalCharges"] = data["TotalCharges"].astype(float)

            # replace 'No internet service' to No for the following columns
            replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies']
            for i in replace_cols:
                data[i] = data[i].replace({'No internet service': 'No'})
            # replace values
            data["SeniorCitizen"] = data["SeniorCitizen"].replace({1: "Yes", 0: "No"})

        if source == 'bold':
            # convert to float type
            data = data.drop(
                columns=["still_installed", "install_date", "deleted_date", "last_ticket_created_at",
                         "first_ticket_created_at"])

            data["num_months_subscriptions0"] = data["num_months_subscriptions0"].astype(float)

        churn={}
        not_churn = {}

        output = {'data': data, 'minority': churn, 'majority': not_churn}

        return output

