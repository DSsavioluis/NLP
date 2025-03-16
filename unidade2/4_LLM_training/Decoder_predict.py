import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transform_decoder import TransformerLanguageModel, TextDataset, build_vocab


def train_language_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()

            logits = model(input_seq)
            logits = logits.reshape(-1, logits.size(-1))
            target_seq = target_seq.reshape(-1)

            loss = criterion(logits, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def generate_text(model, start_text, vocab, max_len=20, device='cpu'):
    model.eval()
    tokens = start_text.lower().split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    input_seq = indices.copy()

    for _ in range(max_len - len(indices)):
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()
            input_seq.append(next_token)

    inv_vocab = {idx: token for token, idx in vocab.items()}
    return " ".join([inv_vocab.get(idx, "<unk>") for idx in input_seq])


if __name__ == '__main__':
    texts = [
        "o sol nasce no leste e se põe no oeste",
        "a lua brilha intensamente durante a noite",
        "as estrelas iluminam o céu em dias claros",
        "o dia nasce com uma nova esperança",
        "a natureza encanta com sua beleza infinita",
        "o vento sopra e leva as folhas",
        "a chuva cai suave sobre a terra",
        "o mar é profundo e misterioso",
        "as flores desabrocham na primavera",
        "os pássaros cantam alegremente ao amanhecer"
    ]

    vocab = build_vocab(texts, min_freq=1)
    vocab_size = len(vocab)

    max_len = 20
    dataset = TextDataset(texts, vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLanguageModel(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2,
                                     dim_feedforward=512, dropout=0.1, max_len=max_len).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_language_model(model, dataloader, criterion, optimizer, device, num_epochs=100)

    start_text = "o sol"
    generated = generate_text(model, start_text, vocab, max_len=20, device=device)
    print("Texto Gerado:", generated)