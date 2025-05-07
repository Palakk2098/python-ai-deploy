from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import settings

HF_TOKEN = settings.HF_TOKEN
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize these as global variables so they're loaded only once
global_tokenizer = None
global_model = None
global_pipeline = None


def get_model_and_tokenizer():
    global global_tokenizer, global_model, global_pipeline

    if global_tokenizer is None:
        # Load the model using the token
        global_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            use_fast=True,  # Use fast tokenizer
            padding_side="left",  # Better for chat models
        )

        # Load model with CPU optimizations
        global_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",  # Use GPU if available, otherwise CPU
        )

        # Create pipeline once
        global_pipeline = pipeline(
            "text-generation",
            model=global_model,
            tokenizer=global_tokenizer,
            device_map="auto",  # Use GPU if available
        )

    return global_tokenizer, global_model, global_pipeline


def truncate_to_fit_context(content, question, tokenizer, max_length=3500):
    """Truncate content to fit within model context window"""
    # Calculate tokens for prompt components
    system_tokens = len(tokenizer.encode("You are a Question Answering model."))
    question_tokens = len(tokenizer.encode(f"Question: {question}\nAnswer:"))
    formatting_tokens = len(tokenizer.encode("Context:\n\n\n\n"))

    # Calculate remaining tokens for content
    available_tokens = (
        max_length - system_tokens - question_tokens - formatting_tokens - 50
    )  # buffer

    # Truncate content if needed
    content_tokens = tokenizer.encode(content)
    if len(content_tokens) > available_tokens:
        content_tokens = content_tokens[:available_tokens]
        truncated_content = tokenizer.decode(content_tokens, skip_special_tokens=True)
        return truncated_content

    return content


def process_question_with_llama(content, question):
    try:
        # Get or initialize the model, tokenizer and pipeline
        tokenizer, model, llm_pipeline = get_model_and_tokenizer()

        # Truncate content if needed to fit context window
        content = truncate_to_fit_context(content, question, tokenizer)

        # Format with clear delineation between context and question
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful Question Answering assistant. Answer the question based only on the provided context. If the answer cannot be found in the context, say 'I don't have enough information to answer this question.'",
            },
            {
                "role": "user",
                "content": f"Context:\n{content}\n\nQuestion: {question}\n\nProvide a concise and accurate answer based only on the information in the context above.",
            },
        ]

        with torch.inference_mode():
            response = llm_pipeline(
                prompt,
                max_new_tokens=256,  # Allow longer responses when needed
                do_sample=True,  # Enable sampling for more natural responses
                top_p=0.9,  # Nucleus sampling
                temperature=0.3,  # Low but not deterministic
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.1,  # Avoid repetitive text
            )

        # Extract just the assistant's response
        full_response = response[0]["generated_text"]

        # Extract only the assistant's answer using proper response parsing
        if isinstance(full_response, list):
            # For newer HF pipelines that return message format
            for message in full_response:
                if message.get("role") == "assistant":
                    return message.get("content", "")
        else:
            # For text format, find the assistant's response
            # This might need adjustment based on actual output format
            assistant_prefix = "assistant"
            if assistant_prefix in full_response.lower():
                answer_parts = full_response.split(assistant_prefix, 1)[1]
                # Clean up any formatting/metadata
                if "content" in answer_parts:
                    # Try to find content field if it's in JSON-like format
                    import re

                    content_match = re.search(r'content["\s:]+([^}]+)', answer_parts)
                    if content_match:
                        return content_match.group(1).strip('" ')
                return answer_parts.strip()

        # Fallback if parsing fails
        return full_response

    except Exception as e:
        import traceback

        print(f"Error in process_question_with_llama: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing the question: {str(e)}"
