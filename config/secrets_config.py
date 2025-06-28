# config/secrets_config.py

import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENV")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

LUNAR_CRUSH_API_KEY = os.getenv("LUNAR_CRUSH_API_KEY")