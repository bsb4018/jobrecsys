import os
import sys
import re
import pandas as pd
from pandas import DataFrame
from src.exception import JobRecException
from src.logger import logging
from src.utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.constants.file_constants import SCHEMA_FILE_PATH
import warnings
warnings.filterwarnings("ignore")

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise JobRecException(e,sys)
        
    def filter_tags(self,htmlstr):
        try:
            re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I)
            re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I) #Script
            re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I) #style
            re_br=re.compile('<br\s*?/?>')
            re_h=re.compile('</?\w+[^>]*>')
            re_comment=re.compile('<!--[^>]*-->')
            htmlstr = str(htmlstr)
            s=re_cdata.sub('',htmlstr)
            s=re_script.sub('',s)
            s=re_style.sub('',s)
            s=re_br.sub('\n',s)
            s=re_h.sub('',s)
            s=re_comment.sub('',s)
            blank_line=re.compile('\n+')
            s=blank_line.sub('\n',s)
            s=s.replace('\\r'," ")
            s=s.replace('\\t'," ")
            s=s.replace('\n'," ")
            s=s.replace('\\n'," ")
            s = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', s, flags=re.MULTILINE)
            s = re.sub(r'[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', s, flags=re.MULTILINE)
            s = re.sub(r'(www)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', s, flags=re.MULTILINE)
            s = re.sub(r'[0-9a-zA-Z.]+@[0-9a-zA-Z.]', " ", s, flags=re.MULTILINE)
            s = re.sub('\xa0', " ", s, flags=re.MULTILINE)
            s = self.replaceCharEntity(s)
            return s
        
        except Exception as e:
            raise JobRecException(e,sys)
    
    def replace(s,re_exp,repl_string):
        return re_exp.sub(repl_string,s)

    def replaceCharEntity(self,htmlstr):
        try:
            CHAR_ENTITIES={'nbsp':' ','160':' ',
            'lt':'<','60':'<',
            'gt':'>','62':'>',
            'amp':'&','38':'&',
            'quot':'"','34':'"',}

            re_charEntity=re.compile(r'&#?(?P<name>\w+);')
            sz=re_charEntity.search(htmlstr)
            while sz:
                entity=sz.group()
                key=sz.group('name')
                try:
                    htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
                    sz=re_charEntity.search(htmlstr)
                except KeyError:
                    htmlstr=re_charEntity.sub('',htmlstr,1)
                    sz=re_charEntity.search(htmlstr)
            return htmlstr
        
        except Exception as e:
            raise JobRecException(e,sys)
        
    def prepare_all_data(self):
        try:
            logging.info("DATA Validation: Loading Ingested Data...")
            users = pd.read_parquet(self.data_ingestion_artifact.users_file_path)
            users['WorkHistoryCount'] = users['WorkHistoryCount'].replace(120, 12)
            #do job specific work
            logging.info("DATA Validation: Dropping NaN and unrequired Columnns")
            users = users.drop(self._schema_config["user_drop_columns"], axis=1)
            #remove nan rows
            users = users.dropna(how='any')
            unique_users_list = users.UserID.unique().tolist()

            jobs = pd.read_parquet(self.data_ingestion_artifact.jobs_file_path)
            jobs = jobs.dropna(how='any')
            Description = [self.filter_tags(i) for i in jobs.Description.values]
            Requirements = [self.filter_tags(i) for i in jobs.Requirements.values]
            jobs.drop(self._schema_config["jobs_drop_columns"], axis=1, inplace=True)
            jobs["Description"] = Description
            jobs["Requirements"] = Requirements
            unique_jobs_list = jobs.JobID.unique().tolist()

            apps = pd.read_parquet(self.data_ingestion_artifact.apps_file_path)
            apps = apps.drop(self._schema_config["apps_drop_columns"], axis=1)
            apps = apps.dropna(how='any')
            apps = apps[(apps.UserID.isin(unique_users_list)) & (apps.JobID.isin(unique_jobs_list))]

            return users,jobs,apps

        except Exception as e:
            raise JobRecException(e,sys)
        
    def train_test_split(self, users,jobs,apps):
        try:
            logging.info("DATA Validation: Doing Train Test Split")
            train_users = users[users["Split"] == "Train"]
            test_users = users[users["Split"] == "Test"]
            train_apps = apps[apps["Split"] == "Train"]
            test_apps = apps[apps["Split"] == "Test"]
            
            train_users.drop(self._schema_config["final_drop_columns"], axis=1, inplace=True)
            test_users.drop(self._schema_config["final_drop_columns"], axis=1, inplace=True)
            train_apps.drop(self._schema_config["final_drop_columns"], axis=1, inplace=True)
            test_apps.drop(self._schema_config["final_drop_columns"], axis=1, inplace=True)

            #print("Data Validated")
            #print(jobs.head(2))
            #print("--------------")
            #print(train_users.head(2))
            #print("--------------")
            #print(test_users.head(2))
            #print("--------------")
            #print(test_apps.head(2))
            #print("--------------")
            #print(test_apps.head(2))
            #print("--------------")
            
            logging.info("DATA Validation: Saving Validated Data")
            save_jobs_file = self.data_validation_config.valid_jobs_file_name
            dir_path = os.path.dirname(save_jobs_file)
            os.makedirs(dir_path, exist_ok=True)
            #jobs.to_csv(dir_path, index=False)
            jobs.to_parquet(save_jobs_file, engine='fastparquet',index=False)

            #save all three files to ingested folder under artifact in parquet format
            save_users_train_file = self.data_validation_config.valid_train_users_file_name
            dir_path = os.path.dirname(save_users_train_file)
            os.makedirs(dir_path, exist_ok=True)
            #train_users.to_csv(dir_path, index=False)
            train_users.to_parquet(save_users_train_file, engine='fastparquet',index=False)

            #save all three files to ingested folder under artifact in parquet format
            save_users_test_file = self.data_validation_config.valid_test_users_file_name
            dir_path = os.path.dirname(save_users_test_file)
            os.makedirs(dir_path, exist_ok=True)
            #test_users.to_csv(dir_path, index=False)
            test_users.to_parquet(save_users_test_file, engine='fastparquet',index=False)

            save_apps_train_file = self.data_validation_config.valid_train_apps_file_name
            dir_path = os.path.dirname(save_apps_train_file)
            os.makedirs(dir_path, exist_ok=True)
            #train_apps.to_csv(dir_path, index=False)
            train_apps.to_parquet(save_apps_train_file, engine='fastparquet',index=False)

            save_apps_test_file = self.data_validation_config.valid_test_apps_file_name
            dir_path = os.path.dirname(save_apps_test_file)
            os.makedirs(dir_path, exist_ok=True)
            #test_apps.to_csv(dir_path, index=False)
            test_apps.to_parquet(save_apps_test_file, engine='fastparquet',index=False)

        except Exception as e:
            raise JobRecException(e,sys)

        
    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            
            users,jobs,apps = self.prepare_all_data()
            self.train_test_split(users,jobs,apps)
            logging.info("DATA Validation: Storing Data Validation Artifacts...")
            data_validation_artifact = DataValidationArtifact(
                valid_train_users_file_path=self.data_validation_config.valid_train_users_file_name,
                valid_test_users_file_path=self.data_validation_config.valid_test_users_file_name,
                valid_jobs_file_path=self.data_validation_config.valid_jobs_file_name,
                valid_train_apps_file_path=self.data_validation_config.valid_train_apps_file_name,
                valid_test_apps_file_path=self.data_validation_config.valid_test_apps_file_name,
            )
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise JobRecException(e, sys)