import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
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

def load_data(logger, data_path:str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.debug("Data file loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def pre_processing(logger, data:pd.DataFrame, test_size:float, target_column_name:str, text_column_name:str) -> tuple:
    try:
        #dropping unwanted data columns
        data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        #renaming the columns
        data.rename(columns = {'v1': target_column_name, 'v2': text_column_name}, inplace = True)
        #splitting the data in train and test
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=2)
        logger.debug("Data Pre Processing successfull")
        return train_data, test_data
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

def main(logger, data_path, target_column_name, text_column_name, test_size, saving_path, train_data_file_name, test_data_file_name):

    #Calling loading data function
    data = load_data(logger, data_path)

    #Calling pre processing function
    train_data, test_data = pre_processing(logger, data, test_size, target_column_name, text_column_name)

    #Calling file saving function
    save_data(logger, train_data, test_data, saving_path, train_data_file_name, test_data_file_name)

    #logging
    logger.debug("Data Ingestion Phase completed successfully")

if __name__ == "__main__":

    #Parms
    log_dir = "logs"
    name_log = "data_ingestion"
    file_name = "data_ingestion.log"

    #Calling logging function
    logger = logging_func(log_dir, name_log, file_name)

    #Load test_size
    parm_path = "params.yaml"
    params = load_parms(logger, parm_path)
    test_size = params['data_ingestion']['test_size']

    #other parms
    data_path = "experiments\spam.csv"
    target_column_name = "target"
    text_column_name = "text"
    saving_path = os.path.join('data', 'data_ingestion')
    train_data_file_name = 'train_data.csv'
    test_data_file_name = 'test_data.csv'

    #Calling main function
    main(logger, data_path, target_column_name, text_column_name, test_size, saving_path, train_data_file_name, test_data_file_name)