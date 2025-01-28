from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer




# if __name__ == "__main__":
#     data_ingestion = DataIngestion()
#     train_data,test_data = data_ingestion.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

#     model_trainer = ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    
class InitComponents:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.train_data,self.test_data = self.data_ingestion.initiate_data_ingestion()

        self.data_transformation=DataTransformation()
        self.train_arr,self.test_arr,_=self.data_transformation.initiate_data_transformation(self.train_data,self.test_data)

        self.model_trainer = ModelTrainer()
        print(self.model_trainer.initiate_model_trainer(self.train_arr,self.test_arr))

    
if __name__ == "__main__":
    start_component=InitComponents()
    