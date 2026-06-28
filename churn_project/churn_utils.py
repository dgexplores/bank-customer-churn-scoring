import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define canonical column names and groupings
DROP_COLS = ['Year', 'CustomerId', 'Surname']

RAW_NUMERIC_COLS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

ENGINEERED_NUMERIC_COLS = [
    'BalanceSalaryRatio', 'ProductDensity', 'EngagementProduct',
    'AgeTenureInteraction', 'TenureByAge', 'CreditPerAge', 'ZeroBalance'
]

CATEGORICAL_COLS = ['Geography', 'Gender']

TARGET_COL = 'Exited'

def clean_and_normalize_columns(df):
    """
    Standardizes and normalizes column headers of the DataFrame to canonical names.
    This handles variations with spaces, lowercase/uppercase, etc.
    """
    mapping = {
        'year': 'Year',
        'customerid': 'CustomerId',
        'customer id': 'CustomerId',
        'surname': 'Surname',
        'creditscore': 'CreditScore',
        'credit score': 'CreditScore',
        'geography': 'Geography',
        'gender': 'Gender',
        'age': 'Age',
        'tenure': 'Tenure',
        'balance': 'Balance',
        'numofproducts': 'NumOfProducts',
        'num of products': 'NumOfProducts',
        'number of products': 'NumOfProducts',
        'hascrcard': 'HasCrCard',
        'has cr card': 'HasCrCard',
        'isactivemember': 'IsActiveMember',
        'is active member': 'IsActiveMember',
        'estimatedsalary': 'EstimatedSalary',
        'estimated salary': 'EstimatedSalary',
        'exited': 'Exited',
        'churn': 'Exited'
    }
    
    cleaned_cols = {}
    for col in df.columns:
        norm_col = str(col).strip().lower()
        if norm_col in mapping:
            cleaned_cols[col] = mapping[norm_col]
        else:
            cleaned_cols[col] = col
            
    return df.rename(columns=cleaned_cols)

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds engineered features to the dataset.
    This operates on a pandas DataFrame.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Prevent Modifying the original DataFrame
        X_out = X.copy()
        
        # Ensure we have pandas DataFrame
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out, columns=RAW_NUMERIC_COLS + CATEGORICAL_COLS)
            
        # 1. BalanceSalaryRatio = Balance / (EstimatedSalary + 1)
        X_out['BalanceSalaryRatio'] = X_out['Balance'] / (X_out['EstimatedSalary'] + 1)
        
        # 2. ProductDensity = NumOfProducts / (Tenure + 1)
        X_out['ProductDensity'] = X_out['NumOfProducts'] / (X_out['Tenure'] + 1)
        
        # 3. EngagementProduct = IsActiveMember * NumOfProducts
        X_out['EngagementProduct'] = X_out['IsActiveMember'] * X_out['NumOfProducts']
        
        # 4. AgeTenureInteraction = Age * Tenure
        X_out['AgeTenureInteraction'] = X_out['Age'] * X_out['Tenure']
        
        # 5. TenureByAge = Tenure / (Age + 1)
        X_out['TenureByAge'] = X_out['Tenure'] / (X_out['Age'] + 1)
        
        # 6. CreditPerAge = CreditScore / (Age + 1)
        X_out['CreditPerAge'] = X_out['CreditScore'] / (X_out['Age'] + 1)
        
        # 7. ZeroBalance = 1 if Balance == 0 else 0
        X_out['ZeroBalance'] = (X_out['Balance'] == 0).astype(float)
        
        return X_out
