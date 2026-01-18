from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields

# Attempt to load from parent dir if locally not found (Hackathon convenience)
import os
from dotenv import load_dotenv

# Try loading from local .env
if os.path.exists(".env"):
    load_dotenv(".env")
# Try loading from parent .env
elif os.path.exists("../.env"):
    load_dotenv("../.env")

settings = Settings()
