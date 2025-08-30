# llm_client.py
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_MODEL_ID:
    raise ValueError("HF_MODEL_ID not set in .env")

# Load tokenizer and model safely
try:
    if HF_TOKEN and HF_TOKEN.strip():  # Only pass token if non-empty
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_auth_token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID, use_auth_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID)
except Exception as e:
    raise RuntimeError(f"Failed to load model '{HF_MODEL_ID}': {e}")

# Create pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU; use 0 for GPU if available
    max_new_tokens=500,
)

def build_prompt(question: str, chunks: list) -> str:
    """Combine context chunks with the question into a prompt."""
    context = "\n\n".join([c.get("metadata", {}).get("text", "") for c in chunks])
    return f"Answer the question based on the context:\n{context}\n\nQuestion: {question}"

def call_hf_granite(prompt: str) -> str:
    """Call the Hugging Face pipeline with the prompt."""
    try:
        output = generator(prompt, max_new_tokens=500)
        if not output or "generated_text" not in output[0]:
            return "No response generated."
        return output[0]["generated_text"].strip()
    except Exception as e:
        return f"LLM call failed: {e}"
