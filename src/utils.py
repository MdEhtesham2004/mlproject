import sys 
from src.logger import logging
from src.exception import CustomException
import numpy as np 
import pandas as pd
import dill 
import os 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            dill.dump(obj,file)
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e :
        raise CustomException(e,sys)


# def evaluate_models(X_train,y_train,X_test,y_test,models,params):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model_name = list(models.keys())[i]
#             model = list(models.values())[i]
#             para = params[model_name]

#             if model_name == "CatBoost":  # Skip GridSearchCV for CatBoost
#                 logging.info(f"Using CatBoost's internal grid search for {model_name}")
#                 model.grid_search(para, X=X_train, y=y_train)
#                 y_test_pred = model.predict(X_test)
#                 report[model_name] = r2_score(y_test, y_test_pred)
#                 continue


            
#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             # best_model = gs.best_estimator_  # Use the best estimator

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)
            
#             # model.fit(X_train,y_train)
            
            
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train,y_train_pred)
#             test_model_score = r2_score(y_test,y_test_pred)
            
#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e,sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            try:
                # Get the hyperparameters for the model
                param_grid = params.get(model_name, {})

                # If the model has a param grid, we perform GridSearchCV
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, verbose=1, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)  # Set the best parameters found
                    logging.info(f"Best params for {model_name}: {gs.best_params_}")
                else:
                    logging.info(f"No hyperparameters to tune for {model_name}")

                # Fit the model on the training data
                model.fit(X_train, y_train)

                # Predict on both train and test data
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate R-squared scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                # Store the results in the report
                # report[model_name] = {
                #     "train_score": train_model_score,
                #     "test_score": test_model_score
                # }
                report[model_name] = test_model_score
                logging.info(f"Model: {model_name}, Train Score: {train_model_score}, Test Score: {test_model_score}")


            except Exception as e:
                logging.error(f"Error occurred while training model {model_name}: {e}")
                # report[model_name] = {"train_score": None, "test_score": None}
                continue

        # Return the final report with all models' performance
        return report

    except Exception as e:
        logging.error(f"An error occurred in the evaluate_models function: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    