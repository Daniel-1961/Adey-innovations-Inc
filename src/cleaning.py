import pandas as pd
import logging
from typing import Optional

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    def __init__(self, file_path: str):
        """Initialize the class with a dataset file path."""
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        logging.info("DataCleaner initialized with file: %s", file_path)

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info("✅ Data loaded successfully from %s", self.file_path)
        except Exception as e:
            logging.error("❌ Error loading data: %s", e)

    def remove_duplicates(self):
        """Removes duplicate rows from the dataset."""
        if self.data is not None:
            initial_shape = self.data.shape
            self.data.drop_duplicates(inplace=True)
            final_shape = self.data.shape
            logging.info("🔄 Duplicates removed. Rows reduced from %d to %d", initial_shape[0], final_shape[0])
        else:
            logging.error("❌ Data is not loaded. Call `load_data()` first.")

    def correct_data_types(self):
        """Corrects data types (e.g., datetime, numeric)."""
        if self.data is not None:
            # Example: Convert purchase_time to datetime
            if 'purchase_time' in self.data.columns:
                self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'], errors='coerce')
                logging.info("📅 Corrected 'purchase_time' to datetime format.")

            # Example: Convert numerical columns to appropriate types
            numerical_cols = self.data.select_dtypes(include=['object']).columns
            for col in numerical_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                logging.info("🔢 Converted column '%s' to numeric.", col)

            # Handle categorical features as needed (e.g., converting to category)
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.data[col] = self.data[col].astype('category')
                logging.info("🔠 Converted column '%s' to category.", col)

        else:
            logging.error("❌ Data is not loaded. Call `load_data()` first.")

    def save_cleaned_data(self, output_path: str):
        """Saves the cleaned dataset to a new CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            logging.info("✅ Cleaned data saved to %s", output_path)
        else:
            logging.error("❌ No data available to save.")

# Example Usage:
if __name__ == "__main__":
    preprocessor = DataCleaner("Fraud_Data.csv")
    
    # Step 1: Load data
    preprocessor.load_data()

    # Step 2: Remove duplicates
    preprocessor.remove_duplicates()

    # Step 3: Correct data types
    preprocessor.correct_data_types()

    # Step 4: Save cleaned data
    preprocessor.save_cleaned_data("Fraud_Data_Cleaned.csv")
