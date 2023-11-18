import json
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def preprocessing_json(json_path):
    with open(json_path, 'r') as f:
        df = pd.DataFrame(json.load(f))

        # Fix data types
        df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})      # SeniorCitizen to categoric variable
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')     # TotalCharges to operable number
        df['TotalCharges'] = df['TotalCharges'].astype('float64')

        # Find and complete NaN value
        df["TotalCharges"].fillna(0, inplace=True)

        # Previously selected features
        ctg_feats = ['gender',
                     'SeniorCitizen',
                     'Partner',
                     'Dependents',
                     'PhoneService',
                     'MultipleLines',
                     'InternetService',
                     'Contract',
                     'PaperlessBilling',
                     'PaymentMethod']
        
        num_feats = ['tenure',
                     'MonthlyCharges',
                     'TotalCharges']
        
        # Create and apply OneHotEncoder for categorical features, and re-create dataframe
        df_ctg = df[ctg_feats]
        df_num = df[num_feats]
        df_res = df["Churn"]

        enc = OneHotEncoder(sparse=False)
        enc_ctg = enc.fit_transform(df_ctg)

        df_enc = pd.DataFrame(enc_ctg,columns=enc.get_feature_names_out(ctg_feats))
        
        df_end = pd.concat([df_num, df_enc, df_res], axis=1)

        return df_end