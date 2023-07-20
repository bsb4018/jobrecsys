import os,sys
import pandas as pd
from src.exception import JobRecException
from src.logger import logging
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.model.jobrecommender import JobsRecommenderModel

class ModelCreator:
    def __init__(self,):
        try:
            pass
        except Exception as e:
            raise JobRecException(e,sys)
        
    def create_model(candidatedf,querydf):
        try:

            jobs = pd.DataFrame(candidatedf["Title"].unique(), columns=["Title"])
            jobs_tf = tf.data.Dataset.prefetch(tf.data.Dataset.from_tensor_slices(dict(jobs)),buffer_size=tf.data.AUTOTUNE)
            jobs_map = jobs_tf.map(lambda x: {"job_title": x["Title"]})
            job_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
            job_titles_vocabulary.adapt(jobs_map.map(lambda x: x["job_title"]))
            job_model = tf.keras.Sequential([job_titles_vocabulary,\
                                             tf.keras.layers.Embedding(job_titles_vocabulary.vocabulary_size(), 64)])
                        
            users_tf = tf.data.Dataset.prefetch(tf.data.Dataset.from_tensor_slices(dict(querydf)), buffer_size=tf.data.AUTOTUNE)
            users_map = users_tf.map(lambda x: {"user_id": x["UserID"]})
            user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
            user_ids_vocabulary.adapt(users_map.map(lambda x: x["user_id"]))
            user_model = tf.keras.Sequential([user_ids_vocabulary,\
                                              tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 64)])
            
            jobs_x = jobs_tf.map(lambda x: x["Title"])
            # Define your objectives.
            task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(jobs_x.batch(128).map(job_model)))
            model = JobsRecommenderModel(user_model, job_model, task)

            return model

        except Exception as e:
            raise JobRecException(e,sys)