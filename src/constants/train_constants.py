import os

SAVED_MODEL_DIR = os.path.join("saved_models")
PIPELINE_NAME: str = "job_rec_sys"
ARTIFACT_DIR: str = "artifact"


'''
Data Ingestion related constant start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_USERS_FILE_NAME: str = "users.parquet"
DATA_INGESTION_JOBS_FILE_NAME: str = "jobs.parquet"
DATA_INGESTION_APPS_FILE_NAME: str = "apps.parquet"
DATA_INGESTION_WINDOW_NUMBER: int = 6


'''
Data Validation related constant start with DATA_VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_TRAIN_USERS_FILE_NAME: str = "valid_train_users.parquet"
DATA_VALIDATION_TEST_USERS_FILE_NAME: str = "valid_test_users.parquet"
DATA_VALIDATION_JOBS_FILE_NAME: str = "valid_jobs.parquet"
DATA_VALIDATION_TRAIN_APPS_FILE_NAME: str = "valid_train_apps.parquet"
DATA_VALIDATION_TEST_APPS_FILE_NAME: str = "valid_test_apps.parquet"


'''
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "transformed_train_apps_data.parquet"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "transformed_test_apps_data.parquet"
DATA_TRANSFORMATION_JOBS_FILE_NAME: str = "transformed_jobs.parquet"


'''
Model Trainer related constant start with MODEL_TRAINER VAR NAME
'''
MODEL_TRAINER_DIR_NAME: str = "model_training"
MODEL_TRAINER_MODEL_WEIGHTS_DIR_NAME: str = "model_saves"
MODEL_TRAINER_MODEL_CHECKPOINTS_DIR_NAME: str = "model_checkpoints"
MODEL_TRAINER_MODEL_MAPS_DIR_NAME: str = "model_data_maps"
MODEL_TRAINER_MODEL_WEIGHTS_FILE_NAME: str = "job_rec_weights"
MODEL_TRAINER_MODEL_CHECKPOINTS_FILE_NAME: str = "job_rec_checkpoints"
MODEL_TRAINER_EPOCHS: int = 5

'''
Model Evaluation related constant start with MODEL_EVALUATION VAR NAME
'''
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_MODEL_REPORT_FILE_NAME: str = "model_report.json"
MODEL_EVALUATION_THRESHOLD_SCORE: str = 0.005

'''
Model Pusher ralated constant start with MODEL_PUSHER VAR NAME
'''
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR