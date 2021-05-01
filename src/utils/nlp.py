import torch.nn as nn

from transformers import AutoTokenizer

class TextToPositionalEncoding(nn.Module):
    """
        Takes in a piece of text and converts it to the form 
        needed by the transformer language model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TextToPositionalEncoding, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = tokenizer.encode(x)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def rawToString(tokenized): 
    return AutoTokenizer.from_pretrained('bert-base-cased').decode(tokenized)