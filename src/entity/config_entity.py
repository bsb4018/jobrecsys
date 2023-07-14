from datetime import datetime
import os
from src.constants.train_constant import *

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = PIPELINE_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.users_file_name: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_USERS_FILE_NAME
        )
        self.jobs_file_name: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_JOBS_FILE_NAME
        )
        self.apps_file_name: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_APPS_FILE_NAME
        )
        self.data_window = DATA_INGESTION_WINDOW_NUMBER


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_valdiation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_train_users_file_name: str = os.path.join(
            self.data_valdiation_dir, DATA_VALIDATION_TRAIN_USERS_FILE_NAME
        )
        self.valid_test_users_file_name: str = os.path.join(
            self.data_valdiation_dir, DATA_VALIDATION_TEST_USERS_FILE_NAME
        )
        self.valid_jobs_file_name: str = os.path.join(
            self.data_valdiation_dir, DATA_VALIDATION_JOBS_FILE_NAME
        )
        self.valid_train_apps_file_name: str = os.path.join(
            self.data_valdiation_dir, DATA_VALIDATION_TRAIN_APPS_FILE_NAME
        )
        self.valid_test_apps_file_name: str = os.path.join(
            self.data_valdiation_dir, DATA_VALIDATION_TEST_APPS_FILE_NAME
        )