from collections import Counter
import torch


# 1. Construindo o vocabulário com uma tokenização simples (split por espaço)
def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)
    # Adiciona tokens especiais
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def preprocess_text(text, vocab, max_len):
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    # Aplica padding se necessário ou trunca a sequência
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)