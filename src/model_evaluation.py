import pandas as pd
import os
import numpy as np
import logging
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

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

def load_model(logger, model_path:str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def load_data(logger, data_path:str) -> pd.DataFrame:
    try:
        #read the data
        test_data = pd.read_csv(data_path)
        logger.debug("Test data loaded successfully")
        return test_data
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def model_evaluation(logger, model, test_data:pd.DataFrame) -> dict:
    try:
        #segregating X_test, y_test
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1]

        #Predicting using model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        #model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        #creating dictionary
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        #logger
        logger.debug("Model evaluation parameters generated")
        return metrics_dict
    except Exception as e:
        logger.error(f"Exception raised : {e}")

def save_dict(logger, saving_path:str, file_name_to_save:str, metrics_dict:dict) -> None:
    try:
        # Ensure the directory exists
        os.makedirs(saving_path, exist_ok=True)

        #path
        path = os.path.join(saving_path, file_name_to_save)
        
        #Saving metrics
        with open(path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)

        #logger
        logger.debug("Metrics saved successfully")

    except Exception as e:
        logger.error(f"Exception raised : {e}")

def main(log_dir, name_log, file_name, model_path, data_path, saving_path, file_name_to_save):

    #logging
    logger = logging_func(log_dir, name_log, file_name)

    #load model
    model = load_model(logger, model_path)

    #load data
    test_data = load_data(logger, data_path)

    #Model evaluation
    metrics_dict = model_evaluation(logger, model, test_data)

    #Save dictionary
    save_dict(logger, saving_path, file_name_to_save, metrics_dict)

    #loggging
    logger.debug("Model Evaluation phase completed successfully")

if __name__ == "__main__":
    log_dir = "logs"
    name_log = "model_evaluation"
    file_name = "model_evaluation.log"
    model_path = os.path.join("model", "model.pkl")
    data_path = os.path.join("data", "feature_engineering", "test_data.csv")
    saving_path = os.path.join("reports")
    file_name_to_save = "metrics_data.json"
    main(log_dir, name_log, file_name, model_path, data_path, saving_path, file_name_to_save) 
