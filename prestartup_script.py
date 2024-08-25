import logging
import os
import sys
import time
from pathlib import Path


logging.info(f"BizyAirEnhancer starts...")


current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
logging.info(f"{current_path} append to PYTHONPATH")
