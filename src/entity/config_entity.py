from datetime import datetime
import os
from src.constants.train_constants import *

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


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_applications_file_name: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRAIN_FILE_NAME
        )
        self.transformed_test_applications_file_name: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TEST_FILE_NAME
        )
        self.transformed_jobs_file_name: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_JOBS_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_training_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.saved_model_dir: str = os.path.join(
            self.model_training_dir, MODEL_TRAINER_MODEL_WEIGHTS_DIR_NAME
        )
        self.saved_checkpoint_dir: str = os.path.join(
            self.model_training_dir, MODEL_TRAINER_MODEL_CHECKPOINTS_DIR_NAME
        )
        self.saved_model_weights_file_path: str = os.path.join(
            self.saved_model_dir, MODEL_TRAINER_MODEL_WEIGHTS_FILE_NAME
        )
        self.saved_model_checkpoints_file_path: str = os.path.join(
            self.saved_checkpoint_dir, MODEL_TRAINER_MODEL_CHECKPOINTS_FILE_NAME
        )
        self.model_epochs: int = MODEL_TRAINER_EPOCHS


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
        )
        self.model_report_file_path: str = os.path.join(
            self.model_evaluation_dir, MODEL_EVALUATION_MODEL_REPORT_FILE_NAME
        )
        self.model_eval_threshold_score: str = MODEL_EVALUATION_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
        )
        #timestamp = round(datetime.now().timestamp())
        self.saved_model_path = os.path.join(SAVED_MODEL_DIR, training_pipeline_config.timestamp, MODEL_PUSHER_PRODUCTION_MODEL_FILE_NAME)
        self.saved_model_data_path = os.path.join(SAVED_MODEL_DIR, training_pipeline_config.timestamp, MODEL_PUSHER_PRODUCTION_MODEL_DATA_DIR)