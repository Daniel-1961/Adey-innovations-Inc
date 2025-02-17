import pandas as pd

def convert_ip_to_int(df):
    """Converts IP addresses to integer format."""
    df['ip_address'] = df['ip_address'].astype(int)
    return df

def merge_with_geolocation(df, geo_df):
    """Merges Fraud_Data.csv with IpAddress_to_Country.csv based on IP address range."""
    geo_df['lower_bound_ip_address'] = geo_df['lower_bound_ip_address'].astype(int)
    geo_df['upper_bound_ip_address'] = geo_df['upper_bound_ip_address'].astype(int)
    
    def find_country(ip):
        match = geo_df[(geo_df['lower_bound_ip_address'] <= ip) & (geo_df['upper_bound_ip_address'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'
    
    df['country'] = df['ip_address'].apply(find_country)
    return df
