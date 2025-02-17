import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_numerical_features(df, columns, method="standard"):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.
    
    Parameters:
        df (DataFrame): The input DataFrame
        columns (list): List of numerical columns to scale
        method (str): Scaling method - "standard" (default) or "minmax"

    Returns:
        DataFrame: Scaled DataFrame
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid method! Use 'standard' or 'minmax'.")

    df[columns] = scaler.fit_transform(df[columns])
    return df


