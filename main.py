from memory_profiler import profile
from src.pipe.train_pipeline import TrainPipeline
import logging
import warnings
warnings.filterwarnings("ignore")

# create logger
logger = logging.getLogger('memory_profile_log')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler("memory_profile.log")
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)

from memory_profiler import LogFile
import sys
sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)

@profile
def main():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == "__main__":
    main()