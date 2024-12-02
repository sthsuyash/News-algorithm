import torch
import pickle
import numpy as np
from src.models.lstm_binary import LSTMBinaryModel
from src.nepali_sentiment_analysis.preprocessing import preprocess_text, tokenizer

# Model parameters
input_dim = 300     # Embedding dimension
hidden_dim = 128    # Number of LSTM units
output_dim = 1      # Number of classes (0 or 1)
num_layers = 2      # Number of LSTM layers
max_length = 32     # Maximum length of the review

labels_dict = {
    0: 'Negative',
    1: 'Positive'
}

# Load the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Word2Vec model
with open('src/models/nepali_word2vec_model.pkl', 'rb') as f:
    word2vec_model = pickle.load(f)

model_file = 'src/models/outputs/lstm_binary_model.pth'

# Load the model
model = LSTMBinaryModel(
    input_dim,
    hidden_dim,
    output_dim,
    num_layers
).to(device)
# model.load_state_dict(torch.load(model_file))
model.load_state_dict(
    torch.load(
        model_file,
        map_location=torch.device('cpu'),
        weights_only=True
    )
)
model.eval()

# Predict on a sample sentence
sample_sentence = "यो फोन खराब छ"
sample_sentence = preprocess_text(sample_sentence)
sample_tokens = tokenizer.tokenize(sample_sentence)
sample_review_vector = []
for token in sample_tokens:
    if token in word2vec_model:
        sample_review_vector.append(word2vec_model[token])
    else:
        sample_review_vector.append(np.zeros(word2vec_model.vector_size))

# Pad the review to max_length with zero vectors
if len(sample_review_vector) > max_length:
    sample_review_vector = sample_review_vector[:max_length]
else:
    sample_review_vector.extend(
        [np.zeros(word2vec_model.vector_size)] * (max_length - len(sample_review_vector)))

sample_review_vector = torch.tensor(
    sample_review_vector,
    dtype=torch.float32
).unsqueeze(0).to(device)
output = model(sample_review_vector)
predicted = (output > 0.5).item()
probability = torch.sigmoid(output).item()


print(f"Sentence: {sample_sentence}")
print(f"Predicted Sentiment: {labels_dict[predicted]}")
print(f"Probability: {probability}")
