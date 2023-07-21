import sys
from src.exception import JobRecException
from src.logger import logging
from src.components.model_recommender import ModelRecommender
from src.model.model_resolver import ModelResolver


class PredictionPipeline:
    def __init__(self,):
        try:
            self.get_recommender = ModelRecommender()
        except Exception as e:
            raise JobRecException(e, sys)
        
    def make_recommendation(self):
        try:
            recommendations_list = self.get_recommender.get_recommendations()
            print(f"Top recommendations for given user : {recommendations_list}")
            return ""
        
        except Exception as e:
            raise JobRecException(e, sys)
        

if __name__ == "__main__":
    model_recommender = PredictionPipeline()
    print(model_recommender.make_recommendation())