import torch
import math
import torch.nn as nn
from nltk.tokenize import RegexpTokenizer

from transformers import AutoTokenizer
from torchtext.vocab import GloVe

class TextToPositionalEncoding(nn.Module):
    """
        Takes in a piece of text and converts it to the form 
        needed by the transformer language model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TextToPositionalEncoding, self).__init__()

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.embedder = GloVe(dim=300)
        self.dimension_change = nn.Linear(300, d_model)

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f"Test sentence is: '{x}'")
        tokens = self.tokenizer.tokenize(x)
        vectors = self.embedder.get_vecs_by_tokens(tokens, lower_case_backup=True)
        vectors = self.dimension_change(vectors)

        # Positional encoding
        input_vecs = vectors + self.pe[:vectors.size(0), :]
        return self.dropout(input_vecs)


def rawToString(tokenized):
    return AutoTokenizer.from_pretrained('bert-base-cased').decode(tokenized)