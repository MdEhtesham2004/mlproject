import os 
import sys 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "XGB": XGBRegressor(use_label_encoder=False, eval_metric="mlogloss"),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            params = {
            "RandomForest": {
                # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "DecisionTree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter': ['best', 'random'],
                # 'max_features': ['sqrt', 'log2'],
            },
            "LinearRegression": {},
            "KNeighbors": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, 0.5, .001],
                # 'loss': ['linear', 'square', 'exponential'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "GradientBoost": {
                # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # 'criterion': ['squared_error', 'friedman_mse'],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "XGB": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoost": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            } # keeping null for catboost to solve the incompatiblily issue
            # "CatBoost": {
            #     'depth': [6, 8, 10],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'iterations': [30, 50, 100]
            # }
            }
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            values = [val for val in model_report.values()]
            best_model_score = max(values)
            



            # best_model_score = max(sorted(model_report.values()))
            # best_model_score = max([model['test_score'] for model in model_report.values()])
            # best_model_name = max(model_report, key=model_report.get)  # This will give you the model name with the highest score
            # best_model_score = model_report[best_model_name]  # This gives you the best score


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Model report: {model_report}")
            logging.info(f"{best_model_name} is the best model with score: {best_model_score} and type:{type(best_model),{type(best_model_score)},{type(best_model_name)}}")
            if best_model_score < 0.6:
                raise CustomException("no best model found!",sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2= r2_score(y_test,predicted)

            return r2



        except Exception as e:
            raise CustomException(e,sys)