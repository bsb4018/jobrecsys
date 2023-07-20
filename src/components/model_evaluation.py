import os,sys
import pandas as pd
from src.exception import JobRecException
from src.logger import logging
from src.entity.artifact_entity import (DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact)
from src.entity.config_entity import ModelEvaluationConfig
import tensorflow as tf
from src.utils import write_json_file
from src.model.model_resolver import ModelResolver
from src.model.model_creator import ModelCreator
import warnings
warnings.filterwarnings("ignore")
  

class ModelEvaluation:
    def __init__(self,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise JobRecException(e,sys) 


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")

            test_file_path = self.data_transformation_artifact.transformed_test_apps_file_path
            jobs_file_path = self.data_transformation_artifact.transformed_jobs_file_path
            
            logging.info("Loading Test Data")
            #Load transformed data
            testdf = pd.read_parquet(test_file_path)
            jobsdf = pd.read_parquet(jobs_file_path)

            logging.info("Converting Test Data to Tensorflow dataset")
            test = tf.data.Dataset.from_tensor_slices(dict(testdf))
            test = tf.data.Dataset.prefetch(test, buffer_size=tf.data.AUTOTUNE)

            testmodel = ModelCreator.create_model(jobsdf,testdf)
            testmodel.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

            logging.info("Loading Model Weights")
            testmodel.load_weights(self.model_trainer_artifact.saved_weights_file_path)

            cached_test = test.batch(4096).cache()
            
            logging.info("Evaluating Model")
            testmodel.evaluate(cached_test, return_dict=True)

            logging.info("Getting Model Scores")            
            top100_accuracy = testmodel.get_metrics_result()['factorized_top_k/top_100_categorical_accuracy'].numpy().item()
            top50_accuracy = testmodel.get_metrics_result()['factorized_top_k/top_50_categorical_accuracy'].numpy().item()
            top10_accuracy = testmodel.get_metrics_result()['factorized_top_k/top_10_categorical_accuracy'].numpy().item()

            #print("Top 100 Accuracy", top100_accuracy)
            #print("Top 50 Accuracy", top50_accuracy)
            #print("Top 10 Accuracy", top10_accuracy)

            #create json report file
            model_eval_report = {
                "top_100_accuracy" : top100_accuracy,
                "top_50_accuracy" : top50_accuracy,
                "top_10_accuracy" : top10_accuracy
            }
            
            write_json_file(self.model_evaluation_config.model_report_file_path, model_eval_report)

            train_model_file_path = self.model_trainer_artifact.saved_weights_directory_path
            model_resolver = ModelResolver()
            is_model_accepted = True
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_score=None,
                    current_model_weights_path = train_model_file_path,
                    current_model_report_file_path = self.model_evaluation_config.model_report_file_path
                )
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            logging.info("Loading and Getting Best Model Scores")
            best_model = ModelCreator.create_model(jobsdf,testdf)
            best_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
            best_model_weights_path = model_resolver.get_best_model_path()
            best_model.load_weights(best_model_weights_path)
            best_model.evaluate(cached_test, return_dict=True)
            
            
            best_model_top_100_accuracy = best_model.get_metrics_result()['factorized_top_k/top_100_categorical_accuracy'].numpy()
            test_model_top_100_accuracy = testmodel.get_metrics_result()['factorized_top_k/top_100_categorical_accuracy'].numpy()
            improved_score = best_model_top_100_accuracy - test_model_top_100_accuracy
            
            logging.info("Evaluating if Current Model is Best Model")
            if self.model_evaluation_config.model_eval_threshold_score < improved_score:
                is_model_accepted=True
            else:
                is_model_accepted=False
            
            logging.info("Saving Model Evaluation Artifacts")
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_score=improved_score,
                    current_model_weights_path = train_model_file_path,
                    current_model_report_file_path = self.model_evaluation_config.model_report_file_path
                    )
            
           
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise JobRecException(e,sys)