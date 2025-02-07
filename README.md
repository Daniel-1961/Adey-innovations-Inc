Fraud Detection - Data Analysis & Preprocessing
Overview
This phase focuses on preparing transaction data for fraud detection modeling through comprehensive data cleaning, analysis, and feature engineering.

Datasets
Fraud_Data.csv: E-commerce transactions
IpAddress_to_Country.csv: IP-country mapping
creditcard.csv: Bank transactions (PCA-transformed features)
Data Preprocessing Steps
1. Handle Missing Values
# Check missing values
print(df.isnull().sum())

# Impute or drop based on analysis
df['age'].fillna(df['age'].median(), inplace=True)
2. Data Cleaning
Remove duplicates: df.drop_duplicates(inplace=True)
Fix data types:
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])
3. Exploratory Data Analysis (EDA)
Univariate Analysis:

plt.figure(figsize=(10,6))
sns.histplot(data=fraud_data, x='purchase_value', hue='class', bins=30)
plt.title('Purchase Value Distribution by Fraud Status')
Bivariate Analysis:

pd.crosstab(fraud_data['source'], fraud_data['class']).plot(kind='bar')
4. Geolocation Merging
# Convert IP to integer
fraud_data['ip_int'] = fraud_data['ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Merge datasets
merged_df = pd.merge_asof(
    fraud_data.sort_values('ip_int'),
    ip_country.sort_values('lower_bound_ip_address'),
    left_on='ip_int',
    right_on='lower_bound_ip_address',
    direction='forward'
)
5. Feature Engineering
Transaction Patterns:

# Transaction frequency
user_activity = fraud_data.groupby('user_id')['purchase_time'].agg(['count', 'nunique'])
Time-Based Features:

fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
6. Normalization & Encoding
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Scale numerical features
scaler = MinMaxScaler()
fraud_data[['purchase_value', 'age']] = scaler.fit_transform(fraud_data[['purchase_value', 'age']])

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_source = encoder.fit_transform(fraud_data[['source']])
