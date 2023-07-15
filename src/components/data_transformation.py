import os
import sys
import pandas as pd
from pandas import DataFrame
from src.exception import JobRecException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from src.constants.file_constants import *
import warnings
warnings.filterwarnings("ignore")


class DataIngestion:
    def __init__(self, data_validation_artifact = DataValidationArtifact, data_transformation_config = DataTransformationConfig):
        try:
            self.data_transofrmation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise JobRecException(e,sys)
        

    def merge_data(self,):
        try:
            #load train test data

            #merge train users and jobs then merge result with merge train apps

            #merge test users and jobs then merge result with merge test apps

            #save the files to location
            pass
        except Exception as e:
            raise JobRecException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            
            self.merge_data()
            
            data_transformation_artifact = DataTransformationArtifact(
                valid_transformed_train_file_path = self.data_transofrmation_config.valid_transform_train_file_name,
                valid_transformed_train_file_path = self.data_transofrmation_config.valid_transform_test_file_name
            )
            logging.info(f"Data ingestion artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise JobRecException(e, sys)