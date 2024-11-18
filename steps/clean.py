import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    def clean_data(self, data):
        data.drop(['CustomerID'], axis=1, inplace=True)
        data["Churn"]= data["Churn"].fillna(1)
        
        df1 = data.copy()
        df1.dropna(inplace=True)
        df1.columns = [col.replace(' ','_') for col in df1.columns]
        categorical_features = df1.select_dtypes('object').columns.tolist()
        df1 = pd.get_dummies(df1, columns=categorical_features, drop_first=True, dtype='int')
        
        
        df1 = df1[['Age', 'Tenure', 'Usage_Frequency', 'Support_Calls', 'Payment_Delay',
        'Total_Spend', 'Last_Interaction','Gender_Male',
        'Subscription_Type_Premium', 'Subscription_Type_Standard',
        'Contract_Length_Monthly', 'Contract_Length_Quarterly', 'Churn']]
        
        numerical_features = df1.select_dtypes('float').columns.tolist()

        numerical_features = list(df1.iloc[:,:-6])
        scaler = StandardScaler()

        for feature in numerical_features:
            df1[feature] = scaler.fit_transform(df1[[feature]])
                
        
        return df1