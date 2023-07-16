import os
import sys
import pandas as pd
from pandas import DataFrame
from src.exception import JobRecException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
import warnings
warnings.filterwarnings("ignore")


class DataTransformation:
    def __init__(self, data_validation_artifact = DataValidationArtifact, data_transformation_config = DataTransformationConfig):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise JobRecException(e,sys)
        

    def merge_data(self,):
        try:
            logging.info("DATA Transformation: Loading data for transformation")
            #load train test data
            #train_usersdf = pd.read_parquet(self.data_validation_artifact.valid_train_users_file_path)
            #test_usersdf = pd.read_parquet(self.data_validation_artifact.valid_test_users_file_path)
            jobs_df = pd.read_parquet(self.data_validation_artifact.valid_jobs_file_path)
            train_appsdf = pd.read_parquet(self.data_validation_artifact.valid_train_apps_file_path)
            test_appsdf = pd.read_parquet(self.data_validation_artifact.valid_test_apps_file_path)

            #xjobs = jobs_df[["JobID","Title","Full_Description"]]
            
            tjobs = jobs_df[["JobID","Title"]]
            #merge train users and jobs then merge result with merge train apps
            trainapps = train_appsdf[["UserID", "JobID"]]
            trainapps = pd.merge(tjobs, trainapps, on='JobID')

            logging.info("DATA Transformation: Merging Required Features")
            #merge test users and jobs then merge result with merge test apps
            testapps = test_appsdf[["UserID", "JobID"]]
            testapps = pd.merge(tjobs, testapps, on='JobID')

            trainapps['JobID'] = trainapps['JobID'].astype(str)
            trainapps['UserID'] = trainapps['UserID'].astype(str)
            testapps['JobID'] = testapps['JobID'].astype(str)
            testapps['UserID'] = testapps['UserID'].astype(str)
            tjobs['JobID'] = tjobs['JobID'].astype(str)

            #print("Data Transformed")
            #print("--------------")
            #print(trainapps.head(2))
            #print("--------------")
            #print(testapps.head(2))
            #print("--------------")
            #print(tjobs.head(2))
            #print("--------------")

            logging.info("DATA Transformation: Saving Transformed Features")
            #save the files to location
            transformed_apps_train_file = self.data_transformation_config.transformed_train_applications_file_name
            dir_path = os.path.dirname(transformed_apps_train_file)
            os.makedirs(dir_path, exist_ok=True)
            #trainapps.to_csv(dir_path, index=False)
            trainapps.to_parquet(transformed_apps_train_file, engine='fastparquet',index=False)

            transformed_apps_test_file = self.data_transformation_config.transformed_test_applications_file_name
            dir_path = os.path.dirname(transformed_apps_test_file)
            os.makedirs(dir_path, exist_ok=True)
            #testapps.to_csv(dir_path, index=False)
            testapps.to_parquet(transformed_apps_test_file, engine='fastparquet',index=False)

            transformed_jobs_file = self.data_transformation_config.transformed_jobs_file_name
            dir_path = os.path.dirname(transformed_jobs_file)
            os.makedirs(dir_path, exist_ok=True)
            #xjobs.to_csv(dir_path, index=False)
            tjobs.to_parquet(transformed_jobs_file, engine='fastparquet',index=False)

        except Exception as e:
            raise JobRecException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            self.merge_data()
            logging.info("DATA Transformation: Storing Data Transformation Artifacts...")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_apps_file_path = self.data_transformation_config.transformed_train_applications_file_name,
                transformed_test_apps_file_path = self.data_transformation_config.transformed_test_applications_file_name,
                transformed_jobs_file_path = self.data_transformation_config.transformed_jobs_file_name
            )
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise JobRecException(e, sys)