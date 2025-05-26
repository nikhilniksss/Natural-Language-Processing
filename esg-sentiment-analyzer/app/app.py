import streamlit as st
from transformers import BertForSequenceClassification
import torch
import torch.nn.functional as F
from utils import preprocess

model = BertForSequenceClassification.from_pretrained('model')
model.eval()

st.title("ESG Sentiment Analyser")
st.write("**Paste any ESG-related financial news, article snippet, or statement below to analyze its sentiment.**")

text = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits,dim = -1)
        pred = torch.argmax(probs,dim=-1).items()
        label_map = {0: "❌ Negative", 1: "⚪ Neutral", 2: "✅ Positive"}
        st.success(f"**Sentiment: {label_map[pred]}**")