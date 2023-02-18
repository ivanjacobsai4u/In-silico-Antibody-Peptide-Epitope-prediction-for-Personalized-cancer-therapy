import logging
import warnings
warnings.filterwarnings("ignore")
from ModelTrainer import ModelTrainer
import json


with open('configs/configs.json',"r") as f:
    configs=json.load(f)

names_models=configs['train']["names_models"]
type_to_implementation =configs['train']["type_to_implementation"]
csv_path_patient_1=configs['train']["csv_path_patient_1"]
csv_path_patient_2=configs['train']["csv_path_patient_2"]

nr_epochs=configs['train']['nr_epochs']
model_trainer=ModelTrainer(names_models=names_models,
                           type_to_implementation=type_to_implementation,
                           csv_path_patient_1=csv_path_patient_1,
                           csv_path_patient_2=csv_path_patient_2)
train_results=model_trainer.train(nr_epochs=nr_epochs,persist_results=True)
logging.info(train_results)
logging.info('finished')