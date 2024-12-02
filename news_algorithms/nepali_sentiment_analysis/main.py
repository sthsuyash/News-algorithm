import torch.nn.functional as F
from gensim.models import Word2Vec
import torch
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
import torch.nn as nn
import math

nltk.download('stopwords')
not_stopwords = {
    'न',          # not
    'नजिकै',     # nearby
    'नत्र',       # otherwise
    'नयाँ',       # new (can be negative in some contexts)
    'नै',         # indeed (can be used in negative context)
    'निम्न',      # low, inferior
    'निम्नानुसार',  # as below
    'बिरुद्ध',    # against
    'बाहेक',      # except
    'गैर',        # non
    'नै',         # non
    'भए',         # became (can imply a negative state)
    'नै',         # no
    'दुई',        # two (context can be negative in some cases)
    'नै',         # non
    'न',          # no
}
stop_words = set(stopwords.words('nepali')) - not_stopwords

# Assuming you want to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x * math.sqrt(self.d_model) + self.pe[:x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, q, k, v):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads,
                                  self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads,
                                  self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads,
                                  self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v)
        concat = scores.transpose(1, 2).contiguous().view(
            bs, -1, self.num_heads * self.d_k)
        return self.out(concat)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class SentimentAnalysisModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_classes, d_ff, max_len, dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# Load the Word2Vec model and other necessary items
with open('models/nepali_word2vec_model.pkl', 'rb') as f:
    word2vec_model = pickle.load(f)  # Your saved Word2Vec model

# Load your trained model and move it to the device (GPU or CPU)
with open('models/sentiment_transformer.pkl', 'rb') as f:
    model = pickle.load(f)  # Your trained model
model.to(device)  # Ensure the model is on the correct device


def preprocess(text: str):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    letters_to_normalize = {
        "ी": "ि",
        "ू": "ु",
        "श": "स",
        "ष": "स",
        "व": "ब",
        "ङ": "न",
        "ञ": "न",
        "ण": "न",
        "ृ": "र",
        "ँ": "",
        "ं": "",
        "ः": "",
        "ं": ""
    }
    for l1, l2 in letters_to_normalize.items():
        text = text.replace(l1, l2)
    return text


def text_to_embeddings(text, word2vec_model, max_length=32):
    tokens = text.split()
    review_vector = []

    for token in tokens:
        if token in word2vec_model:
            review_vector.append(word2vec_model[token])
        else:
            review_vector.append(np.zeros(word2vec_model.vector_size))

    if len(review_vector) > max_length:
        review_vector = review_vector[:max_length]
    else:
        review_vector.extend(
            [np.zeros(word2vec_model.vector_size)] * (max_length - len(review_vector)))

    return np.array(review_vector)


def predict(text, model, word2vec_model):
    preprocessed_text = preprocess(text)
    input_embeddings = text_to_embeddings(preprocessed_text, word2vec_model)

    input_tensor = torch.tensor(
        input_embeddings, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    if output.size(1) == 1:  # Binary classification
        predicted_class = torch.round(output).item()  # 0 or 1
        # Sigmoid output for confidence
        confidence = torch.sigmoid(output).item()
    else:  # Multi-class classification
        predicted_class = torch.argmax(
            output, dim=1).item()  # Index of max probability
        # Max probability as confidence
        confidence = torch.max(F.softmax(output, dim=1)).item()

    class_label = "Positive" if predicted_class == 1 else "Negative"

    return class_label, confidence


input_text = "म राम्रो छैन"
predicted_class, confidence = predict(input_text, model, word2vec_model)
print(f"Predicted Class: {predicted_class} with Confidence: {confidence:.4f}")
