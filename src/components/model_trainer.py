import os,sys
import pandas as pd
from src.exception import JobRecException
from src.logger import logging
from src.entity.artifact_entity import (DataTransformationArtifact,ModelTrainerArtifact)
from src.entity.config_entity import ModelTrainerConfig
import tensorflow as tf
from src.model.model_creator import ModelCreator
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise JobRecException(e,sys) 


    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")

            train_file_path = self.data_transformation_artifact.transformed_train_apps_file_path
            jobs_file_path = self.data_transformation_artifact.transformed_jobs_file_path
            
            logging.info("Loading Train Data...")
            #Load transformed data
            traindf = pd.read_parquet(train_file_path)
            jobsdf = pd.read_parquet(jobs_file_path)
            
            logging.info("Convert loaded data to tensorflow dataset")
            train = tf.data.Dataset.from_tensor_slices(dict(traindf))
            train = tf.data.Dataset.prefetch(train, buffer_size=tf.data.AUTOTUNE)
            
            logging.info("Building the Model")
            model = ModelCreator.create_model(jobsdf,traindf)
            model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
            
            model_checkpoints_filepath = self.model_trainer_config.saved_model_checkpoints_file_path
            #dir_path = os.path.dirname(model_checkpoints_filepath)
            #os.makedirs(dir_path, exist_ok=True)
            model_checkpoint=tf.keras.callbacks.ModelCheckpoint(model_checkpoints_filepath,save_weights_only=True)
            
            logging.info("Started Model Training")
            cached_train = train.shuffle(100_000).batch(8192).cache()
            model.fit(cached_train, epochs=self.model_trainer_config.model_epochs, callbacks=[model_checkpoint], verbose=1)
            logging.info("Model Training Successfull")
                         

            logging.info("Saving Model Weights")
            filepath = self.model_trainer_config.saved_model_weights_file_path
            model.save_weights(filepath=filepath, save_format='tf')
            
            logging.info("Saving Model Training Artifacts")
            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                saved_weights_directory_path = self.model_trainer_config.saved_model_dir,
                saved_weights_file_path = self.model_trainer_config.saved_model_weights_file_path
            )

            logging.info("Exiting initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifact

        except Exception as e:
            raise JobRecException(e,sys)