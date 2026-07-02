from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    MINIO_ENDPOINT:   str   = "localhost:9000"
    MINIO_ACCESS_KEY: str   = "minioadmin"
    MINIO_SECRET_KEY: str   = "minioadmin"
    MINIO_BUCKET:     str   = "signature-poc"
    MINIO_SECURE:     bool  = False

   
    JWT_SECRET:       str   = "change-this-in-production"
    JWT_ALGORITHM:    str   = "HS256"
    JWT_EXPIRY_HOURS: int   = 24

   
    CLASSIFIER_PATH:  str   = "models/document_classifier.pt"
    DETECTOR_PATH:    str   = "models/signature_yolov8_v3.pt"
    CONFIDENCE:       float = 0.7

  
    API_HOST:         str   = "0.0.0.0"
    API_PORT:         int   = 8000

    class Config:
        env_file = ".env"


config = Settings()