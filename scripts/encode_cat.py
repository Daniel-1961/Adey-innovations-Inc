from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical_features(df, columns, method="label"):
    """
    Encodes categorical features using Label Encoding or One-Hot Encoding.
    
    Parameters:
        df (DataFrame): The input DataFrame
        columns (list): List of categorical columns to encode
        method (str): Encoding method - "label" (default) or "onehot"

    Returns:
        DataFrame: Encoded DataFrame
    """
    if method == "label":
        encoder = LabelEncoder()
        for col in columns:
            df[col] = encoder.fit_transform(df[col])
    
    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=True)  # One-hot encoding
    
    else:
        raise ValueError("Invalid method! Use 'label' or 'onehot'.")
    
    return df
