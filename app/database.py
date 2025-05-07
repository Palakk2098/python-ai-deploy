from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# MongoDB connection
client = AsyncIOMotorClient(settings.MONGODB_URL)
database = client[settings.MONGO_DB_NAME]
