import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import TrainingPipelineConfig,DataValidationConfig,DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact

from src.exception import JobRecException
from src.logger import logging
from memory_profiler import profile

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    @profile
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            
            logging.info(
              "Entered the start_data_ingestion method of TrainPipeline class"
            )
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            
            return data_ingestion_artifact
    
        except Exception as e:
            raise JobRecException(e, sys)

    @profile    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
      
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            
            return data_validation_artifact

        except Exception as e:
            raise JobRecException(e,sys)

    @profile    
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
      
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Performed the data transformation operation")
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            
            return data_transformation_artifact

        except Exception as e:
            raise JobRecException(e,sys)
    
    @profile
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()


            logging.info("Performed the Model Training operation")
            logging.info(
                "Exited the start_model_trainer method of TrainPipeline class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise JobRecException(e,sys)
    
    @profile
    def start_model_evaluation(self,data_transformation_artifact:DataTransformationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            logging.info("Entered the start_model_evaluation method of TrainPipeline class")
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_transformation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()

            logging.info("Performed the Model Evaluation operation")
            logging.info(
                "Exited the start_model_evaluation method of TrainPipeline class"
            )
            return model_eval_artifact
        
        except Exception as e:
            raise JobRecException(e,sys)

    @profile    
    def start_model_pusher(self, data_transformation_artifact: DataTransformationArtifact, model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            logging.info("Entered the start_model_pusher method of TrainPipeline class")
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, data_transformation_artifact, model_evaluation_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Performed the Model Pusher operation")
            logging.info(
                "Exited the start_model_pusher method of TrainPipeline class"
            )

            return model_pusher_artifact
        
        except Exception as e:
            raise JobRecException(e,sys)
    

    def run_pipeline(self,) -> None:
        try:
            
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            TrainPipeline.is_pipeline_running=True
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            #data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            #model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            #model_eval_artifact = self.start_model_evaluation(data_transformation_artifact, model_trainer_artifact)
            #if not model_eval_artifact.is_model_accepted:
                #print("Process Completed Succesfully. Model Trained and Evaluated but the Trained model is not better than the best model. So, we do not push this model to Production. Exiting.")
                #raise Exception("Process Completed Succesfully. Model Trained and Evaluated but the Trained model is not better than the best model. So, we do not push this model to Production. Exiting.")
            #model_pusher_artifact = self.start_model_pusher(data_transformation_artifact,model_eval_artifact)
            TrainPipeline.is_pipeline_running=False
              
            logging.info("Training Pipeline Running Operation Complete")
            logging.info(
                "Exited the run_pipeline method of TrainPipeline class"
            )
        except Exception as e:
            TrainPipeline.is_pipeline_running=False
            raise JobRecException(e, sys)