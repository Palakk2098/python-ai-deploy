import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents
from app.middleware import register_middleware

app = FastAPI()


register_middleware(app)

app.include_router(documents.router, prefix="/api", tags=["documents"])


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
