from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-small"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print("Model loaded successfully!")
except Exception as e:
    print("Model load failed:", e)
