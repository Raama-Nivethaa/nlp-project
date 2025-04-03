from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.logger.log_helper import logging
from src.exception.exception import customexception

class ModelEvaluation:
    @staticmethod
    def evaluate(y_test, y_pred, model_name):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)

            logging.info(f"{model_name} Evaluation Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            print(f"{model_name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise customexception(e)