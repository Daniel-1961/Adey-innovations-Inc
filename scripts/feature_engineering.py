import pandas as pd

def calculate_transaction_features(df):
    """Calculate transaction frequency and velocity features."""
    df = df.sort_values(by=['user_id', 'purchase_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count') # transaction frequency
    df['time_since_last_purchase'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() # transaction velocity
    df['time_since_last_purchase'] = df['time_since_last_purchase'].fillna(0)
    return df

def add_time_features(df):
    """Extract hour of day and day of week from purchase time.""" #extraction of time_based features
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df
