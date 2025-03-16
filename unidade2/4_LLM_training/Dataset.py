import torch
from torch.utils.data import Dataset

# 2. Dataset customizado
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        # Padding ou truncamento
        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)