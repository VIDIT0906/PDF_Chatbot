import logging
from typing import List, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_name: str):
    """Load specified Hugging Face LLM locally."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        logging.info(f"LLM {model_name} loaded.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"LLM loading failed: {e}")
        raise

def create_prompt_template(retrieved_chunks: List[str], user_query: str) -> str:
    intro = (
        "You are an AI assistant. "
        "Answer the user's question ONLY using the provided context from documents below. "
        "If the answer is not present in the context, reply: 'I could not find an answer in the documents.'"
    )
    context_str = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    prompt = (
        f"{intro}\n\n"
        f"Context:\n{context_str}\n\n"
        f"User question: {user_query.strip()}\n"
        f"Answer:"
    )
    return prompt

def generate_response_stream(llm_model, tokenizer, prompt: str, max_new_tokens=512) -> Iterator[str]:
    """Streams the LLM response token by token or by sentence."""
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Yield the answer part only (after the prompt)
    answer_start = output_text.find("Answer:")
    if answer_start != -1:
        answer = output_text[answer_start + len("Answer:"):].strip()
    else:
        answer = output_text
    # Stream by sentence
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    partial = ""
    for s in sentences:
        partial += s + " "
        yield partial.strip()
