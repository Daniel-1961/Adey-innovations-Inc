import pandas as pd
import numpy as np
from pathlib import Path

def load_data(credit_path: str, fraud_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate input datasets"""
    credit_df = pd.read_csv(credit_path)
    fraud_df = pd.read_csv(fraud_path, parse_dates=['transaction_timestamp'])
    
    # Validate required columns
    required_credit_cols = {'card_id', 'credit_limit', 'current_balance', 
                           'transaction_count', 'account_age_days', 'last_payment_date'}
    required_fraud_cols = {'credit_card_id', 'transaction_amount', 'billing_country',
                          'ip_country', 'merchant_category', 'transaction_timestamp'}
    
    if not required_credit_cols.issubset(credit_df.columns):
        missing = required_credit_cols - set(credit_df.columns)
        raise ValueError(f"Missing columns in credit data: {missing}")
        
    if not required_fraud_cols.issubset(fraud_df.columns):
        missing = required_fraud_cols - set(fraud_df.columns)
        raise ValueError(f"Missing columns in fraud data: {missing}")
    
    return credit_df, fraud_df

def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for credit card data"""
    df = df.copy()
    
    # Payment behavior features
    df['payment_ratio'] = df['total_payment'] / (df['min_payment_due'] + 1e-6)
    df['credit_utilization'] = df['current_balance'] / df['credit_limit']
    
    # Transaction patterns
    df['txn_freq_per_day'] = df['transaction_count'] / df['account_age_days']
    df['avg_txn_amount'] = df['total_spent'] / df['transaction_count']
    
    # Temporal features
    df['days_since_last_payment'] = (pd.to_datetime('now') - pd.to_datetime(df['last_payment_date'])).dt.days
    
    return df

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for fraud data"""
    df = df.copy()
    
    # Transaction patterns
    df['amount_to_avg'] = df.groupby('credit_card_id')['transaction_amount'].transform(
        lambda x: x / x.mean()
    )
    
    # Geographic features
    df['country_mismatch'] = (df['billing_country'] != df['ip_country']).astype(int)
    
    # Temporal features
    df['hour_of_day'] = df['transaction_timestamp'].dt.hour
    df['is_weekend'] = df['transaction_timestamp'].dt.weekday >= 5
    
    return df

def merge_and_create_features(credit_df: pd.DataFrame, fraud_df: pd.DataFrame) -> pd.DataFrame:
    """Merge datasets and create final features"""
    merged = pd.merge(
        credit_df,
        fraud_df,
        left_on='card_id',
        right_on='credit_card_id',
        how='inner'
    )
    
    # Combined features
    merged['limit_usage_ratio'] = merged['transaction_amount'] / merged['credit_limit']
    merged['recent_activity_flag'] = (
        (merged['days_since_last_payment'] < 7) & 
        (merged['transaction_amount'] > 1000)
    ).astype(int)
    
    return merged

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed data to parquet format"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Data saved successfully to {output_path}")

if __name__ == "__main__":
    # INPUT PATHS (MODIFY THESE)
    CREDIT_DATA_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_creditcard.csv" 
    FRAUD_DATA_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_Fraud_Data.csv"
    OUTPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/feature_engineered_data.csv"
    
    # Pipeline execution
    print("Loading data...")
    credit_data, fraud_data = load_data(CREDIT_DATA_PATH, FRAUD_DATA_PATH)
    
    print("Engineering credit features...")
    credit_features = engineer_credit_features(credit_data)
    
    print("Engineering fraud features...")
    fraud_features = engineer_fraud_features(fraud_data)
    
    print("Merging datasets...")
    final_df = merge_and_create_features(credit_features, fraud_features)
    
    print("Saving results...")
    save_processed_data(final_df, OUTPUT_PATH)
    
    print("Feature engineering complete!")
