import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.logger.log_helper import logging
from src.exception.exception import customexception

# Save and Load Functions
def save_object(file_path, obj):
    """Save an object using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error in saving object: {str(e)}")
        raise customexception(str(e))

def load_object(file_path):
    """Load an object using pickle."""
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error in loading object: {str(e)}")
        raise customexception(str(e))

# Model Evaluation Function
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Train and evaluate multiple models."""
    try:
        model_scores = {}  # Dictionary to store model performance
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Store results
            model_scores[model_name] = accuracy  # Store accuracy score

            # Log and print results
            logging.info(f"{model_name} Evaluation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return model_scores  # Return model performance dictionary
    
    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        raise customexception(str(e))
