import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CreditCardEDA:
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully from {self.file_path}")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")

    def plot_fraud_distribution(self):
        """Plots fraud vs. non-fraud transactions."""
        if self.data is not None:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=self.data["Class"])
            plt.title("Fraud vs Non-Fraud Transactions")
            plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
            plt.show()
        else:
            logging.error("❌ Data is not loaded.")

    def correlation_heatmap(self):
        """Plots correlation heatmap for numerical features."""
        if self.data is not None:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data.corr(), cmap="coolwarm", annot=False)
            plt.title("Feature Correlation Heatmap")
            plt.show()
        else:
            logging.error("❌ Data is not loaded.")




logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IPGeolocationEDA:
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully from {self.file_path}")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")

    def plot_country_distribution(self, top_n=10):
        """Plots distribution of top N most common countries."""
        if self.data is not None:
            country_counts = self.data["country"].value_counts().head(top_n)
            plt.figure(figsize=(10, 5))
            sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis")
            plt.xlabel("Count")
            plt.ylabel("Country")
            plt.title(f"Top {top_n} Countries in Transactions")
            plt.show()
        else:
            logging.error("❌ Data is not loaded.")



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FraudDataEDA:
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully from {self.file_path}")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")

    def summary_statistics(self):
        """Displays summary statistics."""
        if self.data is not None:
            logging.info("📊 Summary Statistics:")
            print(self.data.describe())
        else:
            logging.error("❌ Data is not loaded. Call `load_data()` first.")

    def plot_numerical_distributions(self):
        """Plots histograms and boxplots for numerical features."""
        if self.data is not None:
            numerical_cols = self.data.select_dtypes(include=["number"]).columns
            for col in numerical_cols:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                sns.histplot(self.data[col], kde=True, bins=30)
                plt.title(f"Distribution of {col}")

                plt.subplot(1, 2, 2)
                sns.boxplot(x=self.data[col])
                plt.title(f"Boxplot of {col}")

                plt.show()
        else:
            logging.error("❌ Data is not loaded.")

    def plot_categorical_distributions(self):
        """Plots count plots for categorical features."""
        if self.data is not None:
            categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns
            for col in categorical_cols:
                plt.figure(figsize=(8, 5))
                sns.countplot(x=self.data[col])
                plt.title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                plt.show()
        else:
            logging.error("❌ Data is not loaded.")

    def plot_fraud_distribution(self):
        """Plots fraud vs. non-fraud transactions."""
        if self.data is not None:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=self.data["class"])
            plt.title("Fraud vs Non-Fraud Transactions")
            plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
            plt.show()
        else:
            logging.error("❌ Data is not loaded.")


    

if __name__ == "__main__":
    #credit_eda
    credit_eda = CreditCardEDA("creditcard.csv")
    credit_eda.load_data()
    credit_eda.plot_fraud_distribution()
    credit_eda.correlation_heatmap()

    #ip_eda 
    ip_eda = IPGeolocationEDA("IpAddress_to_Country.csv")
    ip_eda.load_data()
    ip_eda.plot_country_distribution()


    #fraud_eda
    fraud_eda = FraudDataEDA("Fraud_Data.csv")
    fraud_eda.load_data()
    fraud_eda.summary_statistics()
    fraud_eda.plot_numerical_distributions()
    fraud_eda.plot_categorical_distributions()
    fraud_eda.plot_fraud_distribution()
