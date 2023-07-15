from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    users_file_path: str
    jobs_file_path: str
    apps_file_path: str
    
@dataclass
class DataValidationArtifact:
    valid_train_users_file_path: str
    valid_test_users_file_path: str
    valid_train_apps_file_path: str
    valid_test_apps_file_path: str
    valid_jobs_file_path: str


@dataclass
class DataTransformationArtifact:
    valid_transformed_train_file_path: str
    valid_transformed_test_file_path: str