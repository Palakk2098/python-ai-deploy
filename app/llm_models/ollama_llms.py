import requests
from app.config import settings

OLLAMA_MODEL = "openhermes"  # or "mistral"


def process_question_with_ollama(
    content: str, question: str, model_name: str = OLLAMA_MODEL
) -> str:
    """Use local Ollama model to answer a question based on the PDF context."""
    prompt = f"Context:\n{content}\n\nQuestion: {question}\nAnswer:"

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # We want full response, not streaming
        },
        timeout=300,  # Adjust as needed
    )

    response.raise_for_status()
    answer = response.json()["response"].strip()
    return answer
