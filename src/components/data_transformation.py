import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from src.logger.log_helper import logging
from src.exception.exception import customexception
from src.utils.utils import save_object

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    vectorizer_path: str = os.path.join('artifacts', 'vectorizer.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'label_encoder.pkl')


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()

    def preprocess_text(self, text):
        try:
            text = text.lower()
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]

            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

            return ' '.join(tokens)
        except Exception as e:
            logging.error(f"Error in preprocessing text: {str(e)}")
            raise customexception(e)

    def transform_data(self, train_df, test_df):
        try:
            if 'review' not in train_df.columns or 'review' not in test_df.columns:
                raise customexception("Column 'review' not found in dataset.")

            train_df['processed_text'] = train_df['review'].apply(self.preprocess_text)
            test_df['processed_text'] = test_df['review'].apply(self.preprocess_text)

            X_train = self.vectorizer.fit_transform(train_df['processed_text'])
            X_test = self.vectorizer.transform(test_df['processed_text'])

            y_train = self.label_encoder.fit_transform(train_df['sentiment'])
            y_test = self.label_encoder.transform(test_df['sentiment'])

            # Save vectorizer and label encoder
            save_object(self.config.vectorizer_path, self.vectorizer)
            save_object(self.config.label_encoder_path, self.label_encoder)

            logging.info("Data Transformation Completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise customexception(e)
