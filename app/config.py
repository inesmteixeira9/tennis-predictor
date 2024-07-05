import toml
import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DOCS = toml.load(parent_dir+f'/tennis-predictor-docs.toml')

CONFIG = toml.load(parent_dir+f'/tennis-predictor-configs.toml')
APP_NAME = CONFIG['env']['APP_NAME']
APP_VERSION = CONFIG['env']['APP_VERSION']
API_PREFIX = CONFIG['env']['API_PREFIX']
DEBUG = CONFIG['env']['DEBUG']

SCHEMA = toml.load(parent_dir+f'/tennis-predictor-data-schemas.toml')
