import os
from pydantic_settings import BaseSettings  # Updated import


class Settings:
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "AIDOC"

    # Static folder for PDFs
    STATIC_FOLDER: str = "static/pdfs"

    # FQDN url
    PROJECT_URL: str = "https://127.0.0.1:8000/"

    ORIGINS: list = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://192.168.103.191:3000",
        "https://react-ai-deploy-git-main-palakk2098s-projects.vercel.app/",
    ]
    HF_TOKEN: str = ""
    OLLAMA_URL: str = "http://localhost:11434"
    CHROMA_DB_PATH : str =  os.path.expanduser("~/.chroma_data")
    EMBED_COLLECTION_NAME : str = 'doc_embeddings'

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure the static folder exists
os.makedirs(settings.STATIC_FOLDER, exist_ok=True)
