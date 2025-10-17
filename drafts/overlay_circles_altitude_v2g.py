# placeholder; will be replaced by v2g
import logging
import os
LOG_DIR = "/debugs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "overlay_debug.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG)
logger = logging.getLogger(__name__)
