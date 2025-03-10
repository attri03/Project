import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords    # For stopwords
import string
import numpy as np
from wordcloud import WordCloud

#Downloading required resources
nltk.download('stopwords') 
nltk.download('punkt') 

def logging_func(log_dir:str, name_log:str, file_name:str):

    #Creating the required paths
    log_folder_path = os.path.join(log_dir)
    log_file_path = os.path.join(log_folder_path, file_name)

    #Making directory
    os.makedirs(log_folder_path, exist_ok=True)

    #creating object of logging class
    logger = logging.getLogger(name_log)
    logger.setLevel("DEBUG")

    #Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel("DEBUG")
    console_handler.setFormatter(formatter)

    #File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(formatter)

    #logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def load_data(logger, train_data_path:str, test_data_path:str) -> tuple:

    try:
        #Reading the train and test files
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        #Logging
        logger.debug("Train and test data files loaded successfully")

        return train_data, test_data
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def transform_text(text:str) -> str:

    ps = PorterStemmer()

    # Transform the text to lowercase
    text = text.lower()
    
    # Tokenization using NLTK
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing stop words and punctuation
    text = y[:]
    y.clear()
    
    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    # Join the processed tokens back into a single string
    return " ".join(y)

def preprocessing(logger, data:pd.DataFrame, target_column:str, text_column:str, transformed_text_column:str) -> tuple:
    try:
        #Encoding
        encoder = LabelEncoder()
        data[target_column] = encoder.fit_transform(data[target_column])
        logger.debug("Encoding completed on the data successfully")

        #removing duplicate terms
        data = data.drop_duplicates(keep='first')
        logger.debug("Duplicates removed from the data successfully")

        #applying text transformation on the data
        data = data.copy()  # Prevents SettingWithCopyWarning
        data.loc[:, transformed_text_column] = data[text_column].apply(transform_text)
        logger.debug("Text Transformation on the data completed")

        #Filling the NaN values
        data[transformed_text_column].fillna("", inplace=True)
        logger.debug("NaN values filled")

        return data

    except Exception as e:
        logger.error(f"Exception raised : {e}")

def save_data(logger, processed_train_data:pd.DataFrame, processed_test_data:pd.DataFrame, saving_path:str, train_data_file_name:str, test_data_file_name:str, transformed_text_column:str) -> None:
    try:
        #making the data directory
        os.makedirs(saving_path)
        #saving the training data
        train_path = os.path.join(saving_path, train_data_file_name)
        processed_train_data.to_csv(train_path, index=False)
        #saving the testing data
        test_path = os.path.join(saving_path, test_data_file_name)
        processed_test_data.to_csv(test_path, index=False)
        #logging
        logger.debug("Saved the files successfully")
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def main(log_dir:str, name_log:str, file_name:str, train_data_path:str, test_data_path:str, target_column:str, text_column:str, transformed_text_column:str, saving_path:str, train_data_file_name:str, test_data_file_name:str):
    
    #Calling logging function
    logger = logging_func(log_dir, name_log, file_name)

    #Calling loading data function
    train_data, test_data = load_data(logger, train_data_path, test_data_path)

    #Calling pre processing function
    processed_train_data = preprocessing(logger, train_data, target_column, text_column, transformed_text_column)
    processed_test_data = preprocessing(logger, test_data, target_column, text_column, transformed_text_column)

    #Calling file saving function
    save_data(logger, processed_train_data, processed_test_data, saving_path, train_data_file_name, test_data_file_name, transformed_text_column)

    #logging
    logger.debug("Data PreProcessing Phase completed successfully")

if __name__ == "__main__":

    #Parms
    log_dir = "logs"
    name_log = "data_preprocessing"
    file_name = "data_preprocessing.log"
    train_data_path = os.path.join("data", "data_ingestion", "train_data.csv")
    test_data_path = os.path.join("data", "data_ingestion", "test_data.csv")
    target_column = "target"
    text_column = "text"
    transformed_text_column = "transformed_text"
    saving_path = os.path.join('data', 'data_preprocessing')
    train_data_file_name = "train_data.csv"
    test_data_file_name = "test_data.csv"

    #Calling main function
    main(log_dir, name_log, file_name, train_data_path, test_data_path, target_column, text_column, transformed_text_column, saving_path, train_data_file_name, test_data_file_name)