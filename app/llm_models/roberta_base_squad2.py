from transformers import pipeline

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def process_question_with_rbs(content, question):
    answer = ""
    result = qa_model(question=question, context=content)
    if result and result['answer'].strip():
        answer = result['answer']
    return answer