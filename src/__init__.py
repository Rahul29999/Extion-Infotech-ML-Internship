#!/usr/bin/env python3
"""
Extion Infotech - Machine Learning Internship
Project: Sentiment Analysis on Financial News
Author: Rahul Kumar Sharma
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt

# Load model and tokenizer
MODEL = "yiyanghkust/finbert-tone"
LABELS = ['negative', 'neutral', 'positive']

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Sample financial news headlines
headlines = [
    "Apple stock rises after strong earnings report",
    "Tesla faces investigation over safety concerns",
    "Amazon posts neutral Q2 revenue in line with expectations",
    "Markets tumble as inflation fears grow",
    "Goldman Sachs reports record quarterly profit"
]

# Predict sentiment for each headline
sentiments = []
for text in headlines:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    sentiment = LABELS[torch.argmax(probs)]
    sentiments.append(sentiment)
    print(f"{text} ➤ Sentiment: {sentiment}")

# Plot sentiment distribution
count = Counter(sentiments)
plt.bar(count.keys(), count.values(), color=['red', 'gray', 'green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Headlines")
plt.show()
