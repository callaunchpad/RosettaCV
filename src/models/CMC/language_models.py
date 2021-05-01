import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

import utils.model_io as model_io
import utils.data_io as data_io
from utils.nlp import TextToPositionalEncoding, rawToString
from contrastive_multiview import View

from typing import TypeVar, List, Callable


class TextEncoder(nn.Module):
    def __init__(self, d_input, d_output, nhead=8, num_layers=6):
        self.preprocess = TextToPositionalEncoding(d_input)

        encoder_layer = TransformerEncoderLayer(d_input, nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_input, d_output)    

    def forward(self, x, mask=None, src_key_mask=None):
        x = self.preprocess(x)
        x = self.encoder(x, mask, src_key_mask)
        x = F.ReLU(x)
        x = self.fc1(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, d_input_words, d_input_latent, nhead=8, num_layers=6):

        decoder_layer = TransformerDecoderLayer(d)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc1 = nn.Linear(d_input_latent, d_input_words)

    def forward(self, target, memory, tgt_mask=None, mem_mask=None, tgt_key_mask=None, mem_key_mask=None):
        tgt_mask = tgt_mask if tgt_mask else self.gen_nopeek_mask(target.shape[1])
        
        memory = self.fc1(memory)
        out = self.decoder(target, memory, tgt_mask, mem_mask, tgt_key_mask, mem_key_mask)
        return out

    def gen_nopeek_mask(length):
        mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
