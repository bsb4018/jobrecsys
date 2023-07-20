from typing import Dict, Text
import tensorflow_recommenders as tfrs
import tensorflow as tf

class JobsRecommenderModel(tfrs.Model):
    def __init__(self,user_model: tf.keras.Model,job_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and job representations.
        self.user_model = user_model
        self.job_model = job_model

        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features["UserID"])
        job_embeddings = self.job_model(features["Title"])

        return self.task(user_embeddings, job_embeddings)