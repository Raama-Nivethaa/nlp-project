import os
import sys
import pandas as pd
import joblib
from src.exception.exception import customexception
from src.logger.log_helper import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        print("Initializing NLP Prediction Pipeline...")
    
    def predict(self, text):
        try:
            vectorizer_path = os.path.join("artifacts", "vectorizer.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            vectorizer = load_object(vectorizer_path)
            model = load_object(model_path)
            
            # Transform text input
            transformed_text = vectorizer.transform([text])
            prediction = model.predict(transformed_text)
            
            return prediction[0]
        
        except Exception as e:
            raise customexception(e, sys)

class CustomTextData:
    def __init__(self, text: str):
        self.text = text
    
    def get_data_as_dataframe(self):
        try:
            data_dict = {"Text": [self.text]}
            df = pd.DataFrame(data_dict)
            logging.info("Text DataFrame created successfully.")
            return df
        except Exception as e:
            logging.error("Exception occurred in text data frame creation.")
            raise customexception(e, sys)
