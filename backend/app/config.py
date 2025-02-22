from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    app_name: str = "Job Application Form System"
    cors_origins: List[str] = ["http://localhost:5173", "https://form-data-generator-app-ejyrfp7n.devinapps.com"]  # Allow deployed frontend

settings = Settings()
