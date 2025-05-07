from typing import Dict, Any, Optional

from bson import ObjectId
from pydantic import BaseModel


class Document(BaseModel):
    file_name: str
    file_path: str
    file_size: Any
    file_text_content: str
    file_extracted_details: Dict[str, str]

    class Config:
        # This is to ensure that ObjectId is correctly serialized
        json_encoders = {ObjectId: str}


class DocumentInDB(Document):
    id: str


class FileRequest(BaseModel):
    fileid: str


class QNA(BaseModel):
    fileid: str
    question: Optional[str]
    model_name: str
    is_qna: bool
