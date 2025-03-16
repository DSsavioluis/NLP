import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)

    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab


class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len=50):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        input_seq = torch.tensor(indices[:-1], dtype=torch.long)
        target_seq = torch.tensor(indices[1:], dtype=torch.long)
        return input_seq, target_seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :, :].squeeze(1)



class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt_mask=None):
        x = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)

        output = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        logits = self.fc_out(output)
        return logits