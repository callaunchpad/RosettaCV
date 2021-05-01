import torch

import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, GPT2Config, PretrainedConfig, EncoderDecoderModel, GPT2Tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils.nlp import TextToPositionalEncoding, rawToString


class TextEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        """
        Sets up the encoder with the given latent dimension to encode to
        """
        super(TextEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        # Change dimension of output
        bert_output_size = 768
        self.linear = nn.Linear(bert_output_size, latent_dim)

    def forward(self, x):
        # Tokenize the text
        tokenized = torch.IntTensor(self.tokenizer(x, padding=True)['input_ids'])

        # Forward pass
        model_outputs = self.model(tokenized).last_hidden_state
        # Average over sequence dimension
        model_outputs = torch.mean(model_outputs, dim=-2)
        model_outputs = model_outputs.view(model_outputs.shape[0], -1)

        return F.relu(self.linear(model_outputs))


class TextDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super(TextDecoder, self).__init__()

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt")

        # Define a GPT model as decoder
        dummy_encoder = PretrainedConfig()
        gpt_decoder = GPT2Config()

        # Encoder decoder model
        self.decoder_model = EncoderDecoderModel.from_encoder_decoder_pretrained(dummy_encoder, gpt_decoder)

        decoder_output_size = 512

        self.linear = nn.Linear(decoder_output_size, latent_dim)

    def forward(self, latent_encoding: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        # Embed the decoder input
        decoder_inputs = self.tokenizer(decoder_inputs)
        decoding = self.decoder_model(decoder_input_ids=decoder_inputs, encoder_outputs=latent_encoding)

        return F.relu(self.linear(decoding))


if __name__ == "__main__":
    te = TextEncoder(512)
    print(te(["Testing one two there", "testing here here here here here here here"]).shape)