import os
import sys
import pandas as pd
import numpy as np

from src.logger.log_helper import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, evaluate_model

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle  # For loading the saved model

class ModelTrainerConfig:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression()
        }
        self.best_model = None  # Store the best trained model

    def train_and_select_best_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training models and evaluating performance...")

            # Evaluate models and get performance report
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models=self.models)

            # Print model evaluation report
            print("\nModel Performance Report:\n", model_report)
            logging.info(f"Model Report: {model_report}")

            if not model_report:
                raise ValueError("Model evaluation failed. No valid model performance results.")

            # Get the best model based on accuracy score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            self.best_model = self.models[best_model_name]  # Store best model

            print(f"\nBest Model Found: {best_model_name}, Accuracy Score: {best_model_score}\n")
            logging.info(f"Best Model Found: {best_model_name}, Accuracy Score: {best_model_score}")

            # Ensure directory exists before saving the model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            save_object(file_path=self.config.model_path, obj=self.best_model)

            logging.info(f"Best model saved at {self.config.model_path}")

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(str(e), sys)  

    def predict(self, X_test):
        """Predict using the trained best model"""
        try:
            if self.best_model is None:
                # Load model if it's not in memory
                if os.path.exists(self.config.model_path):
                    with open(self.config.model_path, 'rb') as f:
                        self.best_model = pickle.load(f)
                else:
                    raise ValueError("No trained model found. Train the model first.")

            return self.best_model.predict(X_test)
        
        except Exception as e:
            logging.error(f"Error in model prediction: {str(e)}")
            raise customexception(str(e), sys)
