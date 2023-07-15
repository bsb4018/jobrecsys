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
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "valid_train_data.parquet"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "valid_test_data.parquet"
