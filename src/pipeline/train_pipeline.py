import os
import sys
import pandas as pd
from src.logger.log_helper import logging
from src.exception.exception import customexception

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation


def main():
    try:
        logging.info("🚀 Starting Sentiment Analysis Pipeline")

        # ✅ Step 1: Data Ingestion
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f"✅ Data Ingestion completed. Train Path: {train_data_path}, Test Path: {test_data_path}")

        # ✅ Step 2: Read Data from CSV
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        # ✅ Step 3: Data Transformation
        data_transformation = DataTransformation(config=DataTransformationConfig())
        X_train, X_test, y_train, y_test = data_transformation.transform_data(train_df, test_df)
        logging.info("✅ Data Transformation completed.")

        # ✅ Step 4: Model Training
        model_trainer_obj = ModelTrainer(ModelTrainerConfig())
        model_trainer_obj.train_and_select_best_model(X_train, y_train, X_test, y_test)
        logging.info("✅ Model Training completed.")

        # ✅ Step 5: Get Predictions
        y_pred = model_trainer_obj.predict(X_test)  # ✅ Predict using trained model
        logging.info("✅ Model Prediction completed.")

        # ✅ Step 6: Model Evaluation
        model_eval_obj = ModelEvaluation()
        model_eval_obj.evaluate(y_test, y_pred, "Best Model")
        logging.info("✅ Model Evaluation completed.")

    except Exception as e:
        logging.error(f"❌ Exception occurred during the pipeline execution: {e}")
        raise customexception(e, sys)


if __name__ == "__main__":
    main()
