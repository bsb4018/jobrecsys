import os
import sys
import pandas as pd
from pandas import DataFrame
from src.exception import JobRecException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.file_constants import *
import warnings
warnings.filterwarnings("ignore")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise JobRecException(e,sys)

    def load_data(self):
        try:
            users = pd.read_csv(USERS_FILE_PATH,delimiter='\t',encoding='utf-8')
            users = users[users.WindowID == self.data_ingestion_config.data_window]

            jobs = pd.read_csv(JOBS_FILE_PATH,delimiter='\t',encoding='utf-8',error_bad_lines=False)
            jobs = jobs[jobs.WindowID == self.data_ingestion_config.data_window]
            
            unique_users_list = users.UserID.unique().tolist()
            unique_jobs_list = jobs.JobID.unique().tolist()
            
            applications = pd.read_csv(APPLICATIONS_FILE_PATH, delimiter='\t', encoding='utf-8', error_bad_lines=False)
            applications = applications[(applications.UserID.isin(unique_users_list)) & (applications.JobID.isin(unique_jobs_list))]
            
            save_users_file = self.data_ingestion_config.users_file_name
            dir_path = os.path.dirname(save_users_file)
            os.makedirs(dir_path, exist_ok=True)
            users.to_parquet(dir_path, index=False)

            save_jobs_file = self.data_ingestion_config.jobs_file_name
            dir_path = os.path.dirname(save_jobs_file)
            os.makedirs(dir_path, exist_ok=True)
            jobs.to_parquet(dir_path, index=False)

            save_apps_file = self.data_ingestion_config.apps_file_name
            dir_path = os.path.dirname(save_apps_file)
            os.makedirs(dir_path, exist_ok=True)
            applications.to_parquet(dir_path, index=False)
        
        except Exception as e:
            raise JobRecException(e, sys)
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            self.load_data()
            
            data_ingestion_artifact = DataIngestionArtifact(
                users_file_path=self.data_ingestion_config.users_file_name,
                jobs_file_path=self.data_ingestion_config.jobs_file_name,
                apps_file_path=self.data_ingestion_config.apps_file_name,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise JobRecException(e, sys)