import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    creditcard_df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\10Academy Files\\week_8\\week_8 project\\Adey-innovations-Inc\\data\\creditcard.csv")
    fraud_data_df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\10Academy Files\\week_8\\week_8 project\\Adey-innovations-Inc\\data\\cleaned_Fraud_data.csv")
    return creditcard_df, fraud_data_df

def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    creditcard_df, fraud_data_df = load_data()
    
    # Process credit card dataset
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = preprocess_data(creditcard_df, "Class")
    
    # Process fraud data dataset
    X_train_fd, X_test_fd, y_train_fd, y_test_fd = preprocess_data(fraud_data_df, "class")
    
    # Save processed data
    X_train_cc.to_csv("../data/X_train_cc.csv", index=False)
    X_test_cc.to_csv("../data/X_test_cc.csv", index=False)
    y_train_cc.to_csv("../data/y_train_cc.csv", index=False)
    y_test_cc.to_csv("../data/y_test_cc.csv", index=False)
    
    X_train_fd.to_csv("../data/X_train_fd.csv", index=False)
    X_test_fd.to_csv("../data/X_test_fd.csv", index=False)
    y_train_fd.to_csv("../data/y_train_fd.csv", index=False)
    y_test_fd.to_csv("../data/y_test_fd.csv", index=False)

    print("Data preparation complete and saved.")
