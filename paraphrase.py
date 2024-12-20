# paraphrase.py

import pandas as pd
import re
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the PEGASUS model and tokenizer
model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def generate_paraphrases(sentence):
    inputs = tokenizer([sentence], truncation=True, padding=True, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=60,
        num_return_sequences=2,
        num_beams=5,
        do_sample=True,
        top_k=150,
        top_p=0.9,
        temperature=1.2,
        early_stopping=True
    )
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases

def count_words(phrase):
    words = re.findall(r'\b\w+\b', phrase)
    return len(words)
