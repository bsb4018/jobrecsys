import os,sys
import pandas as pd
import numpy as np
from src.exception import JobRecException
from src.logger import logging
from src.entity.artifact_entity import (DataTransformationArtifact,ModelTrainerArtifact)
from src.entity.config_entity import ModelTrainerConfig
import tensorflow as tf
from model.model_creator import ModelCreator
import tensorflow_recommenders as tfrs
import warnings
warnings.filterwarnings("ignore")

class ModelRecommender:
    def __init__(self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise JobRecException(e,sys) 


    def initiate_model_recommend(self) -> ModelTrainerArtifact:

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")

            train_file_path = self.data_transformation_artifact.transformed_train_apps_file_path
            jobs_file_path = self.data_transformation_artifact.transformed.jobs_file_path
            
            #Load transformed data
            traindf = pd.read_parquet(train_file_path)
            jobsdf = pd.read_parquet(jobs_file_path)

            train = tf.data.Dataset.from_tensor_slices(dict(train))
            train = tf.data.Dataset.prefetch(train, buffer_size=tf.data.AUTOTUNE)

            #test = tf.data.Dataset.from_tensor_slices(dict(test))
            #test = tf.data.Dataset.prefetch(test, buffer_size=tf.data.AUTOTUNE)
            
            model = ModelCreator.create_model(jobsdf,traindf)
            model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

            # Use brute-force search to set up retrieval using the trained representations.
            jobs = pd.DataFrame(jobsdf["Title"].unique(), columns=["Title"])
            jobs_tf = tf.data.Dataset.prefetch(tf.data.Dataset.from_tensor_slices(dict(jobs)),buffer_size=tf.data.AUTOTUNE)
            jobs_x = jobs_tf.map(lambda x: x["Title"])
            
            index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
            index.index_from_dataset(jobs_x.batch(100).map(lambda title: (title, model.job_model(title))))
           
            applied_list = list(traindf[traindf['UserID'] == "433316"].Title)
            # Get some recommendations.
            _, titles = index(np.array(["433316"]))
            recommendations = titles[0, 0:20]
            rec_list = []
            for item in recommendations:
                if item not in applied_list:
                    rec_list.append(item.numpy().decode())
            print(f"Top recommendations for given user : {rec_list}")

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                saved_weights_file_path = self.model_trainer_config.saved_model_weights_file_path
            )
            
            return model_trainer_artifact

        except Exception as e:
            raise JobRecException(e,sys)