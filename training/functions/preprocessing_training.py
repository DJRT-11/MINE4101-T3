import json
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def preprocessing_json(json_path):
    with open(json_path, 'r') as f:
        df = pd.DataFrame(json.load(f))

        # Fix data types
        #df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})      # SeniorCitizen to categoric variable
        df['Dependents'] = df['Dependents'].replace({'Yes': 1, 'No': 0})
        df['PaperlessBilling'] = df['PaperlessBilling'].replace({'Yes': 1, 'No': 0})
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')     # TotalCharges to operable number
        df['TotalCharges'] = df['TotalCharges'].astype('float64')

        # Find and complete NaN value
        df["TotalCharges"].fillna(0, inplace=True)

        # Previously selected features
        ctg_feats = ["Contract",
            "DeviceProtection",
            "InternetService",
            "OnlineBackup",
            "OnlineSecurity",
            "PaymentMethod",
            "StreamingMovies",
            "StreamingTV",
            "TechSupport"]

        num_feats = ['tenure',
                     'MonthlyCharges',
                     'TotalCharges']
        
        oth_feats = ["Dependents",
            "PaperlessBilling",
            "SeniorCitizen",
            "Churn"]
        
        # Create and apply OneHotEncoder for categorical features, and re-create dataframe
        df_ctg = df[ctg_feats]
        df_num = df[num_feats]
        df_oth = df[oth_feats]

        enc = OneHotEncoder(sparse=False)
        enc_ctg = enc.fit_transform(df_ctg)

        df_enc = pd.DataFrame(enc_ctg,columns=enc.get_feature_names_out(ctg_feats))
        
        df_end = pd.concat([df_num, df_enc, df_oth], axis=1)

        cols = df_end.columns.tolist()
        cols.sort()
        df_end = df_end[cols]

        return df_end