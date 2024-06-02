import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch
from torch.nn import functional as F
from main.model import BERT_Arch  

# Load the pre-trained BERT model
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the BERT_Arch model architecture
model = BERT_Arch(bert)

# Load the pre-trained weights
model.load_state_dict(torch.load('saved_weights.pt'))

# Set the model to evaluation mode
model.eval()

# Define the evaluate_message function
def evaluate_message(message):
    # Tokenize and encode the message
    tokens = tokenizer.encode_plus(message, max_length=25, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Make predictions
    with torch.no_grad():
        preds = model(input_ids, attention_mask)
        preds = F.softmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()

    # Interpret the predictions
    result = np.argmax(preds, axis=1)[0]
    print("Predictions:", preds)
    return result


