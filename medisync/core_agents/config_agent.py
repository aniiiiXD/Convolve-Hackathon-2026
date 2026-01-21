from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" 

import os
from dotenv import load_dotenv

# Robust .env loading for the Workspace context
# 1. Current dir
if os.path.exists(".env"):
    load_dotenv(".env")
# 2. Project root (../../)
elif os.path.exists("../../.env"):
    load_dotenv("../../.env")
# 3. User root (fallback)
elif os.path.exists("../.env"):
    load_dotenv("../.env")

settings = Settings()
