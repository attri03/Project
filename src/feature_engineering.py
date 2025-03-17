import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import os
import yaml

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

def load_parms(logger, parm_path:str) -> dict:
    try:
        with open(parm_path, 'r') as file:
            parms = yaml.safe_load(file)
        logger.debug(f"Parms loaded successfully")
        return parms
    except Exception as e:
        logger.error(f"Exception raised : {e}")

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

def feature_eng(logger, train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int, text_column:str, target_column:str) -> tuple:

    try:
        #Filling NaN values in the data
        train_data[text_column].fillna("", inplace=True)
        test_data[text_column].fillna("", inplace=True)

        #Setting the object of TfidfVectorizer
        tfid = TfidfVectorizer(max_features=max_features)

        #transforming X_train, y_train
        X_train = tfid.fit_transform(train_data[text_column]).toarray()
        y_train = train_data[target_column].values

        #transforming X_test, y_test
        X_test = tfid.transform(test_data[text_column]).toarray()
        y_test = test_data[target_column].values

        #Creating train df
        train_df = pd.DataFrame(X_train)
        train_df[target_column] = y_train

        #Creating test df
        test_df = pd.DataFrame(X_test)
        test_df[target_column] = y_test

        #logging
        logger.debug("Feature engineering completed in training and testing data")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Exception raised : {e}")

def save_data(logger, train_data:pd.DataFrame, test_data:pd.DataFrame, saving_path:str, train_data_file_name:str, test_data_file_name:str) -> None:
    try:
        #making the data directory
        os.makedirs(saving_path)
        #saving the training data
        train_path = os.path.join(saving_path, train_data_file_name)
        train_data.to_csv(train_path, index=False)
        #saving the testing data
        test_path = os.path.join(saving_path, test_data_file_name)
        test_data.to_csv(test_path, index=False)
        #logging
        logger.debug("Saved the files successfully")
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def main(logger, train_data_path:str, test_data_path:str, max_features:int, text_column:str, target_column:str, saving_path:str, train_data_file_name:str, test_data_file_name:str):

    #Calling loading data function
    train_data, test_data = load_data(logger, train_data_path, test_data_path)

    #Calling feature engineering function
    engineered_train_data, engineered_test_data = feature_eng(logger, train_data, test_data, max_features, text_column, target_column)

    #Calling file saving function
    save_data(logger, engineered_train_data, engineered_test_data, saving_path, train_data_file_name, test_data_file_name)

    #logging
    logger.debug("Feature Engineering Phase completed successfully")

if __name__ == "__main__":

    #Parms
    log_dir = "logs"
    name_log = "feature_engineering"
    file_name = "feature_engineering.log"

    #Calling logging function
    logger = logging_func(log_dir, name_log, file_name)

    #parms
    parm_path = "params.yaml"
    params = load_parms(logger, parm_path)
    max_features = params['feature_engineering']['max_features']

    #other
    train_data_path = os.path.join("data", "data_preprocessing", "train_data.csv")
    test_data_path = os.path.join("data", "data_preprocessing", "test_data.csv")
    text_column = 'transformed_text'
    target_column = 'target'
    saving_path = os.path.join('data', 'feature_engineering')
    train_data_file_name = "train_data.csv"
    test_data_file_name = "test_data.csv"

    main(logger, train_data_path, test_data_path, max_features, text_column, target_column, saving_path, train_data_file_name, test_data_file_name)