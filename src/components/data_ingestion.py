import os
import sys
import pandas as pd
from pandas import DataFrame
from src.exception import JobRecException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.file_constants import APPLICATIONS_FILE_PATH,JOBS_FILE_PATH,USERS_FILE_PATH
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
            logging.info("DATA INGESTION: Loading Data...")
            users = pd.read_csv(USERS_FILE_PATH,delimiter='\t',encoding='utf-8', on_bad_lines='skip')
            users = users[users.WindowID == self.data_ingestion_config.data_window]

            jobs = pd.read_csv(JOBS_FILE_PATH,delimiter='\t',encoding='utf-8',on_bad_lines='skip')
            jobs = jobs[jobs.WindowID == self.data_ingestion_config.data_window]
            
            unique_users_list = users.UserID.unique().tolist()
            unique_jobs_list = jobs.JobID.unique().tolist()
            
            applications = pd.read_csv(APPLICATIONS_FILE_PATH, delimiter='\t', encoding='utf-8', on_bad_lines='skip')
            applications = applications[(applications.UserID.isin(unique_users_list)) & (applications.JobID.isin(unique_jobs_list))]
            
            #print("Data Ingested")
            #print(users.head(2))
            #print("--------------")
            #print(jobs.head(2))
            #print("--------------")
            #print(applications.head(2))
            #print("--------------")

            logging.info("DATA INGESTION: Creating Artifacts Directory and Saving Required Data...")
            save_users_file = self.data_ingestion_config.users_file_name
            dir_path = os.path.dirname(save_users_file)
            os.makedirs(dir_path, exist_ok=True)
            #users.to_csv(dir_path, index=False)
            logging.info(save_users_file)
            users.to_parquet(save_users_file, engine='fastparquet', index=False)

            save_jobs_file = self.data_ingestion_config.jobs_file_name
            dir_path = os.path.dirname(save_jobs_file)
            os.makedirs(dir_path, exist_ok=True)
            #jobs.to_csv(dir_path, index=False)
            jobs.to_parquet(save_jobs_file, engine='fastparquet', index=False)

            save_apps_file = self.data_ingestion_config.apps_file_name
            dir_path = os.path.dirname(save_apps_file)
            os.makedirs(dir_path, exist_ok=True)
            #applications.to_csv(dir_path, index=False)
            applications.to_parquet(save_apps_file, engine='fastparquet', index=False)
        
        except Exception as e:
            raise JobRecException(e, sys)
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            self.load_data()
            logging.info("DATA INGESTION: Storing Artifacts...")
            data_ingestion_artifact = DataIngestionArtifact(
                users_file_path=self.data_ingestion_config.users_file_name,
                jobs_file_path=self.data_ingestion_config.jobs_file_name,
                apps_file_path=self.data_ingestion_config.apps_file_name,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise JobRecException(e, sys)