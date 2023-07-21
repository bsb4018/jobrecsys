import os,sys
import pandas as pd
import numpy as np
from src.exception import JobRecException
from src.logger import logging
import tensorflow as tf
from src.model.model_creator import ModelCreator
from src.model.model_resolver import ModelResolver
import tensorflow_recommenders as tfrs
import warnings
warnings.filterwarnings("ignore")


class ModelRecommender:
    def __init__(self,):
        try:
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise JobRecException(e,sys) 

    def timestamp_converter(self,integer_timestamp):
        try:
            # Convert the integer back to individual components
            second = integer_timestamp % 100
            integer_timestamp //= 100
            minute = integer_timestamp % 100
            integer_timestamp //= 100
            hour = integer_timestamp % 100
            integer_timestamp //= 100

            year = integer_timestamp % 10000
            integer_timestamp //= 10000
            day = integer_timestamp % 100
            integer_timestamp //= 100
            month = integer_timestamp % 100
            
            # Format the components as strings with leading zeros if necessary
            day_str = str(day).zfill(2)
            month_str = str(month).zfill(2)
            year_str = str(year).zfill(4)
            hour_str = str(hour).zfill(2)
            minute_str = str(minute).zfill(2)
            second_str = str(second).zfill(2)

            # Concatenate the components with underscores
            timestamp = f"{month_str}_{day_str}_{year_str}_{hour_str}_{minute_str}_{second_str}"
            return timestamp
        
        except Exception as e:
            raise JobRecException(e,sys)
        
    def get_recommendations(self,):

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            
            #if not self.model_resolver.is_model_exists():
                #print("No model available. Please train a model first")
                
            best_model_path = self.model_resolver.get_best_model_path()
            vtimestamp = best_model_path.split("\\")[-1]
            ftimestamp = self.timestamp_converter(int(vtimestamp))
            best_model_path = best_model_path.split("\\")[0] + "\\" + ftimestamp
            best_model_weights_path = os.path.join(best_model_path, "production_model", "job_rec_weights")
            best_model_data_path = os.path.join(best_model_path, "production_data")

            
            train_file_path = os.path.join(best_model_data_path,"transformed_train_apps_data.parquet")
            jobs_file_path = os.path.join(best_model_data_path,"transformed_jobs.parquet")
            #Load transformed data
            traindf = pd.read_parquet(train_file_path)
            jobsdf = pd.read_parquet(jobs_file_path)

            train = tf.data.Dataset.from_tensor_slices(dict(traindf))
            train = tf.data.Dataset.prefetch(train, buffer_size=tf.data.AUTOTUNE)

            model = ModelCreator.create_model(jobsdf,traindf)
            model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
            model.load_weights(best_model_weights_path)

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
            #print(f"Top recommendations for given user : {rec_list}")

            return rec_list

        except Exception as e:
            raise JobRecException(e,sys)
        
