import os,sys
import pandas as pd
from src.exception import JobRecException
from src.logger import logging
from src.utils import load_numpy_array_data, load_object,save_object
from src.entity.artifact_entity import (DataTransformationArtifact,ModelTrainerArtifact)
from src.entity.config_entity import ModelTrainerConfig
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
from typing import Dict, Text
import tensorflow_recommenders as tfrs
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

    #def train_model(self, x_train, y_train):
        #try:
            #pass
        #except Exception as e:
            #raise e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")

            train_file_path = self.data_transformation_artifact.transformed_train_apps_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_apps_file_path
            jobs_file_path = self.data_transformation_artifact.transformed.jobs_file_path
            
            #Load transformed data
            traindf = pd.read_parquet(train_file_path)
            testdf = pd.read_parquet(test_file_path)
            jobsdf = pd.read_parquet(jobs_file_path)


            jobs = pd.DataFrame(jobsdf["Title"].unique(), columns=["Title"])
            jobs_tf = tf.data.Dataset.from_tensor_slices(dict(jobs))
            jobs_tf = tf.data.Dataset.prefetch(jobs_tf,buffer_size=tf.data.AUTOTUNE)

            users_tf = tf.data.Dataset.prefetch(tf.data.Dataset.from_tensor_slices(dict(traindf)), buffer_size=tf.data.AUTOTUNE)

           
            users_map = users_tf.map(lambda x: {
            "user_id": x["UserID"]})
            jobs_map = jobs_tf.map(lambda x: {"job_title": x["Title"]})

            job_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
            job_titles_vocabulary.adapt(jobs_map.map(lambda x: x["job_title"]))

            user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
            user_ids_vocabulary.adapt(users_map.map(lambda x: x["user_id"]))

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
            )
            
            return model_trainer_artifact

        except Exception as e:
            raise JobRecException(e,sys)