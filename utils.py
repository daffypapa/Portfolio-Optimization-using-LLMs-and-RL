import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List

def standardize_by_ticker(df: pd.DataFrame, 
                          feature_cols: List[str]):
    df_scaled = df.copy()
    for tic in df['tic'].unique():
        mask = df['tic'] == tic
        scaler = StandardScaler()
        df_scaled.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])
    return df_scaled
