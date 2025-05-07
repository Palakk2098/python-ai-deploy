import logging
import os
import re
import chromadb

from bson import ObjectId
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.database import database
from app.models import Document

collection = database["AIDOC"]
client = chromadb.PersistentClient(settings.CHROMA_DB_PATH)


async def save_document(document_data):
    """
    function to save the documents related details into the mongodb
    """
    document = Document(
        file_name=document_data.file_name,
        file_path=document_data.file_path,
        file_size=document_data.file_size,
        file_text_content=document_data.file_text_content,
        file_extracted_details=document_data.file_extracted_details,
    )
    result = await collection.insert_one(document.dict())
    return document, result.inserted_id


async def get_documents():
    """
    return the documents from the mongodb
    """
    documents = []
    cursor = collection.find(
        {}, {"_id": 1, "file_name": 2, "file_path": 3, "file_size": 4}
    )
    async for doc in cursor:
        documents.append(doc)
    return documents


def convert_objectid_to_str(doc):
    if isinstance(doc, dict):
        # Convert ObjectId fields to string
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)  # Convert ObjectId to string
            elif isinstance(value, dict):  # Recursively handle nested dicts
                convert_objectid_to_str(value)
    return doc


async def get_file_name(fileid):
    """
    function to get the document name from mongodb
    """
    try:
        if isinstance(fileid, str):
            fileid = ObjectId(fileid)
        document = await collection.find_one(
            {"_id": fileid}, {"file_name": 1, "_id": 0}
        )
        if document:
            return document["file_name"]  # Return the document directly
        return None  # If no document is found
    except Exception as e:
        logging.exception(msg=str(e))
        return None


async def set_file_data(fileid, file_data):
    try:
        if isinstance(fileid, str):
            fileid = ObjectId(fileid)
        await collection.update_one(
            {"_id": fileid}, {"$set": {"file_extracted_details": file_data}}
        )
    except Exception as e:
        logging.exception(msg=str(e))


def extract_pdf(file_path):
    """
    function to read the pdf contents using the langchain's PyPDFLoader library
    """
    try:
        loader = PyPDFLoader(file_path=file_path, mode="page", pages_delimiter="")

        docs = []
        docs_lazy = loader.lazy_load()

        for doc in docs_lazy:
            docs.append(doc.page_content)

        return "".join(docs)
    except Exception as e:
        logging.exception(msg=str(e))
        return ""


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
        return "%3.1f %s" % (num, x)


def get_file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)
    return None


async def get_file_content(fileid):
    """
    function to send the saved document content from mongodb
    """
    try:
        if isinstance(fileid, str):
            fileid = ObjectId(fileid)
        document = await collection.find_one(
            {"_id": fileid}, {"_id": 0, "file_name": 1, "file_extracted_details": 2}
        )
        if document.get("file_extracted_details"):
            return " ".join(
                document["file_extracted_details"].split()
            )  # Return the document directly
        else:
            file_location = os.path.join(
                settings.STATIC_FOLDER, document.get("file_name")
            )
            file_content = extract_pdf(file_location)
            await set_file_data(fileid, file_content)
            return " ".join(file_content.split())  # If no document is found
    except Exception as e:
        logging.exception(msg=str(e))
        return None


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def create_chunks(file_content, max_words=1000, overlap=100):

    splitter = SpacyTextSplitter(pipeline="en_core_web_trf", chunk_size=max_words, chunk_overlap=overlap)

    file_content = clean_text(file_content)
    
    chunks = splitter.split_text(file_content)

    return chunks


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks, convert_to_tensor=True).tolist()

    return embeddings

def store_embeddings(chunks,embeddings,doc_id):
    
    chroma_collection = client.get_or_create_collection(name=settings.EMBED_COLLECTION_NAME)
    
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    chroma_collection.add(documents=chunks,embeddings=embeddings,ids=ids,metadatas=metadatas)
    print(chroma_collection.get(),"---documents---")
    return chroma_collection.get()
    


async def process_document_async(document_id: str, file_content: str):
    """
    Background task to process document chunks, create, and store embeddings
    """
    try:
        # Chunking the file content
        chunks = create_chunks(file_content)

        # Creating embeddings from chunks
        embeddings = create_embeddings(chunks)
        
        # Store the embeddings to the vector DB
        store_embeddings(chunks,embeddings,document_id)

    except Exception as e:
        logging.exception(f"Error processing document {document_id}: {str(e)}")
