from src.exception import JobRecException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact,ModelPusherArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
import os,sys
import shutil

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                       data_transformation_artifact: DataTransformationArtifact,
                       model_eval_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_artifact = model_eval_artifact
        except Exception as e:
            raise JobRecException(e,sys)

    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:

            logging.info("Into the initiate_model_pusher function of ModelPusher class")
            trained_model_path = self.model_evaluation_artifact.current_model_weights_path
            
            #Pushing the trained model in the model storage space
            model_file_path = self.model_pusher_config.model_pusher_dir
            shutil.copytree(trained_model_path,model_file_path)
            #os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            #shutil.copy(src=trained_model_path, dst=model_file_path)
            
            logging.info("Saving Model to Production")
            #Pushing the trained model in a the saved path for production
            saved_model_path = self.model_pusher_config.saved_model_path
            shutil.copytree(trained_model_path,saved_model_path)
            #os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)            
            #shutil.copy(src=trained_model_path, dst=saved_model_path)

            logging.info("Saving Model Data to Production")
            #Pushing the trained model data in a the saved path for production
            apps_file_path = self.data_transformation_artifact.transformed_train_apps_file_path
            jobs_file_path = self.data_transformation_artifact.transformed_jobs_file_path

            apps_filename = os.path.basename(apps_file_path)
            jobs_filename = os.path.basename(jobs_file_path)
            #dir_path = os.path.dirname(self.model_pusher_config.saved_model_data_path)
            os.makedirs(self.model_pusher_config.saved_model_data_path, exist_ok=True)
            saved_apps_model_path = os.path.join(self.model_pusher_config.saved_model_data_path,apps_filename)
            saved_jobs_file_path = os.path.join(self.model_pusher_config.saved_model_data_path,jobs_filename)
            shutil.copy(apps_file_path,saved_apps_model_path)
            shutil.copy(jobs_file_path,saved_jobs_file_path)

            
            logging.info("Saving Model Pusher Artifact")
            #Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path, 
                model_file_path=model_file_path)
            
            logging.info(f"Model Pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise JobRecException(e,sys)