import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from src.logger.log_helper import logging
from src.exception.exception import customexception
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(r"C:\Users\neel\OneDrive\Desktop\dataset\movies.csv", encoding='ISO-8859-1')
            logging.info("Reading dataset into DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw dataset in artifacts folder")

            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion process completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error during data ingestion")
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    
    # Step 1: Ingest data and get file paths
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Step 2: Load CSV files as DataFrames
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Step 3: Apply data transformation
    config = DataTransformationConfig()
    data_transformation = DataTransformation(config)

    X_train, X_test, y_train, y_test = data_transformation.transform_data(train_data, test_data)
    
    model_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_config)
    model_trainer.train_and_select_best_model(X_train, y_train, X_test, y_test)
