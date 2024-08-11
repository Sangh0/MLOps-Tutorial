import os

from dotenv import load_dotenv


# load .env file
load_dotenv()


class Config:
    # Authentication
    AUTH_TOKEN = os.getenv("AUTH_TOKEN")

    # Data
    DATA_DIR = os.getenv("DATA_DIR")