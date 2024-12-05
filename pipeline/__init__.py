from pathlib import Path
from libs.data_utils import read_yaml
import logging

PARAMS_FILE_PATH = Path("params.yaml")
PARAMS = read_yaml(PARAMS_FILE_PATH)
ENV = PARAMS.env

# Clear any existing handlers to avoid duplication
logging.getLogger().handlers.clear()

# Configure the file handler
file_handler = logging.FileHandler(f'logs/{ENV}.log', mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure the console (stream) handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get the root logger and add both handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
