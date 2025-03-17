import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
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
        df = pd.read_csv(data_path)
        logger.debug("Loaded data successfully")
        return df
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def train_model(logger, n_estimators:int, random_state:int, train_data:pd.DataFrame) -> RandomForestClassifier:
    try:
        #Converting training data to X_train and y_train
        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1]
        logger.debug("X_train and y_train split successful")
        #Training RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)
        rfc.fit(X_train,y_train)
        logger.debug("Training of RandomForestClassifier successfull")
        return rfc
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def save_model(logger, rfc:RandomForestClassifier, saving_location:str, saving_file_name:str) -> None:
    try:
        #Making the required directory
        os.makedirs(saving_location,exist_ok=True)
        #location
        location_to_save = os.path.join(saving_location, saving_file_name)
        #saving the model
        with open(location_to_save, 'wb') as file:
            pickle.dump(rfc, file)
        logger.debug("Model saved successfully")
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def main(logger, data_path, n_estimators, random_state, saving_location, saving_file_name):

    #loading data
    train_data = load_data(logger, data_path)
    #training model
    rfc = train_model(logger, n_estimators, random_state, train_data)
    #saving model
    save_model(logger, rfc, saving_location, saving_file_name)
    #logger
    logger.debug("Completed the model building phase")

if __name__ == "__main__":

    #Parms
    log_dir = "logs"
    name_log = "model_training"
    file_name = "model_training.log"

    #Calling logging function
    logger = logging_func(log_dir, name_log, file_name)

    #parms
    parm_path = "params.yaml"
    params = load_parms(logger, parm_path)
    n_estimators = params['model_training']['n_estimators']
    random_state = params['model_training']['random_state']

    #others
    data_path = os.path.join("data", "feature_engineering", "train_data.csv")
    saving_location = os.path.join('model')
    saving_file_name = "model.pkl"

    main(logger, data_path, n_estimators, random_state, saving_location, saving_file_name)