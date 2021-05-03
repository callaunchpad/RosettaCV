import torch

import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertLMHeadModel, BertConfig


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
        tokenized = self.tokenizer(x, padding=True, return_tensors="pt")['input_ids']

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
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Decoder model
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True
        config.add_cross_attention = True

        self.decoder_model = BertLMHeadModel.from_pretrained("bert-base-uncased", config=config)

        decoder_input_size = 768
        self.linear = nn.Linear(latent_dim, decoder_input_size)

        # Identifier to signal to the trainer to put the label in the decode call
        self.needs_labels = True

    def forward(self, latent_encoding: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:

        # Change dimension of the input to match cross-attention in BERT
        latent_encoding = F.relu(self.linear(latent_encoding))

        # Tokenize the inputs
        decoder_inputs = self.tokenizer(decoder_inputs, return_tensors="pt", padding=True).input_ids

        # Replicate the latent embedding to mimic an encoder output
        sequence_length = decoder_inputs.size()[1]

        latent_encoding = latent_encoding.unsqueeze(1)
        encoder_output = torch.tile(latent_encoding, (1, sequence_length, 1))

        # Generate logits for prediction
        return self.decoder_model(decoder_inputs, encoder_hidden_states=encoder_output).logits

    def generate(self, latent_encoding: torch.Tensor) -> str:
        """
        Generates a caption for the latent encoding
        :param latent_encoding: The latent encoding to generate from
        :return: The caption generated
        """
        raise NotImplementedError()  # TODO: Implement during evaluation stage
