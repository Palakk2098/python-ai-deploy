from transformers import pipeline

qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def process_question_with_dbcds(content, question):
    answer = ""
    result = qa_model(question=question, context=content)
    if result and result['answer'].strip():
        answer = result['answer']
    return answer