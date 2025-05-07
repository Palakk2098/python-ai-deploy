from transformers import pipeline

qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

def process_question_with_mus(content, question):
    answer = ""
    result = qa_model(question=question, context=content)
    if result and result['answer'].strip():
        answer = result['answer']
    return answer