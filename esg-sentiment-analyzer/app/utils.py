import re
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("model")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "",text)
    text = re.sub(r'[@#]', '', text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def preprocess(text):
    cleaned = clean_text(text)
    return tokenizer(cleaned,return_tensors='pt',padding = True,truncation = True,max_length = 128)

