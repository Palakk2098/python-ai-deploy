import asyncio
import logging
import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.config import settings
from app.crud import save_document, get_documents, convert_objectid_to_str, get_file_name, extract_pdf, get_file_size, \
    get_file_content, process_document_async
from app.llm_models.distilbert_base_cased_distilled_squad import process_question_with_dbcds
from app.llm_models.minilm_uncased_squad2 import process_question_with_mus
from app.llm_models.roberta_base_squad2 import process_question_with_rbs
from app.llm_models.ollama_llms import process_question_with_ollama
from app.llm_models.llama import process_question_with_llama
from app.models import Document, QNA
from app.util import Models


router = APIRouter()


@router.post("/documents/")
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    API to upload the PDFs, save into the static folder, extract the information and save into the mongodb
    """
    try:
        # Check if all files are PDFs
        for file in files:
            if file.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Save the uploaded PDFs to the static folder and insert filenames into MongoDB
        saved_files = []
        for file in files:
            file_location = os.path.join(settings.STATIC_FOLDER, file.filename)

            # Save the file to the static folder
            with open(file_location, "wb") as f:
                f.write(await file.read())

            file_size = get_file_size(file_location)

            file_content = extract_pdf(file_location)

            document_data = Document(file_name=file.filename, file_path=settings.PROJECT_URL + file_location,
                file_size=file_size, file_text_content=file_content, file_extracted_details={}, )
            # Save the file name to MongoDB
            document, inserted_id = await save_document(document_data)

            background_tasks.add_task(
                process_document_async,
                str(inserted_id),
                file_content
            )

            # Store the result
            saved_files.append(
                {"file_name": document.file_name, "file_path": document.file_path, "file_size": document.file_size})
        return {"status": 200, "result": saved_files}

    except Exception as e:
        logging.exception(msg=str(e))
        return {"status": 400, "result": "Error While Adding the File(s)"}


@router.get("/documents/")
async def fetch_documents():
    """
    API to return the documents from the mongodb
    """
    try:
        documents = await get_documents()
        documents = [convert_objectid_to_str(doc) for doc in documents]
        return {"status": 200, "result": documents}
    except Exception as e:
        logging.exception(msg=str(e))
        return {"status": 400, "result": "Files Not Found"}


@router.post("/qna/")
async def question_and_answer(qna: QNA):
    """
    API for extracting the contract information and for the custom question answer.
    """
    try:
        llm_response = ""
        file_content = await get_file_content(qna.fileid)
        if qna.is_qna:
            llm_response = ""
            match qna.model_name:
                case Models.RBS.value:
                    llm_response = process_question_with_rbs(
                        content=file_content, question=qna.question
                    )
                case Models.DBCDS.value:
                    llm_response = process_question_with_dbcds(
                        content=file_content, question=qna.question
                    )
                case Models.MUS.value:
                    llm_response = process_question_with_mus(
                        content=file_content, question=qna.question
                    )
                case Models.OPENHERMES.value:
                    llm_response = process_question_with_ollama(
                        content=file_content, question=qna.question
                    )
                case Models.LLAMA.value:
                    llm_response = process_question_with_llama(
                        content=file_content, question=qna.question
                    )
                case _:
                    return {
                        "status": 400,
                        "result": {"answer": "Select the valid model"},
                    }

            return {"status": 200, "result": {"answer": llm_response}}
        else:
            questions = {"contract_title": "give me the summary of the contract",
                "contract_type": "give me the contract type",
                "contract_start_date": "give me the starting date of the contract",
                "contract_end_date": "give me the ending date of the contract",
                "contract_amount": "give me the amount for the contract",
                "contract_naisc_code": "give me the NAISC code of the contract",
                "contract_eligibility_criteria": "give me the Eligibility Criteria of the contract",
                "contract_scope": "give me the Scope of the contract",
                "contract_processing_for_biding": "Procedure for bidding for the contract"}

            def run_blocking_inference():
                match qna.model_name:
                    case Models.RBS.value:
                        return {
                            key: process_question_with_rbs(
                                content=file_content, question=question
                            )
                            for key, question in questions.items()
                        }
                    case Models.DBCDS.value:
                        return {
                            key: process_question_with_dbcds(
                                content=file_content, question=question
                            )
                            for key, question in questions.items()
                        }
                    case Models.MUS.value:
                        return {
                            key: process_question_with_mus(
                                content=file_content, question=question
                            )
                            for key, question in questions.items()
                        }
                    case Models.OPENHERMES.value:
                        return {
                            key: process_question_with_ollama(
                                content=file_content, question=question
                            )
                            for key, question in questions.items()
                        }
                    case Models.LLAMA.value:
                        return {
                            key: process_question_with_llama(
                                content=file_content, question=question
                            )
                            for key, question in questions.items()
                        }
                    case _:
                        return {
                            "status": 400,
                            "result": {"answer": "Select the valid model"},
                        }

            return await asyncio.to_thread(run_blocking_inference)

    except Exception as e:
        logging.exception(msg=str(e))
        return ''


@router.get("/documents/{fileid}")
async def get_documents_binary(fileid: str):
    """
    API to get the document by id and send into binary format
    """
    try:
        file_name = await get_file_name(fileid)

        file_path = os.path.join(settings.STATIC_FOLDER, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")

        return FileResponse(file_path, media_type="application/pdf", filename=f"{file_name}.pdf")

    except Exception as e:
        logging.exception(msg=str(e))
        return ""
