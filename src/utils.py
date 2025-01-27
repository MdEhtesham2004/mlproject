import sys 
from src.logger import logging
from src.exception import CustomException
import numpy as np 
import pandas as pd
import dill 
import os 


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            dill.dump(obj,file)
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e :
        raise CustomException(e,sys)
