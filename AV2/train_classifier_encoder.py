import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

# ** Hiperparâmetros **

MAX_LEN = 128
EMBED_DIM = 128
VOCAB_SIZE = 4000
NUM_HEADS = 4
NUM_LAYERS = 3
HIDDEN_DIM = 256
DROPOUT = 0.4
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 3e-4

#acuracia = 51%

# ** Carregar dataset e corrigir os rótulos **
df = pd.read_csv("dataset_preprocessed.csv")

# 🔹 Converter os valores da coluna "fake_news" de string para inteiros
df["fake_news"] = df["fake_news"].apply(lambda x: int(eval(x)[0]) if isinstance(x, str) and x.startswith("[") else int(x))

# Contagem das classes antes do balanceamento
print("Contagem de classes antes do balanceamento:")
print(df["fake_news"].value_counts())

# Separar classes corretamente
df_fake = df[df["fake_news"] == 1]
df_real = df[df["fake_news"] == 0]

# 🔹 Se houver classes vazias, não fazer balanceamento
if len(df_fake) > 0 and len(df_real) > 0:
    # Aplicar oversampling na classe minoritária (Fake News)
    df_fake_upsampled = resample(df_fake, replace=True, n_samples=len(df_real), random_state=42)
    df_balanced = pd.concat([df_real, df_fake_upsampled])
else:
    print("⚠️ Aviso: O dataset já está balanceado ou contém apenas uma classe.")
    df_balanced = df

# Embaralhar o dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Verificar contagem de classes após balanceamento
print("Contagem de classes após o balanceamento:")
print(df_balanced["fake_news"].value_counts())

# 🔹 Evitar erro: Se ainda estiver vazio, encerrar a execução
if len(df_balanced) == 0:
    raise ValueError("Erro: O dataset está vazio após balanceamento!")

# 🔹 Dividir dataset corretamente
train_size = int(0.8 * len(df_balanced))
df_train = df_balanced[:train_size]
df_test = df_balanced[train_size:]

# ** Criar vocabulário manualmente **
def build_vocab(texts, vocab_size=VOCAB_SIZE):
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted_vocab[:vocab_size - 2])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

# Criar vocabulário baseado nos títulos
vocab = build_vocab(df_balanced["title"].tolist())

# ** Dataset PyTorch **
class FakeNewsDataset(Dataset):
    def __init__(self, df, text_column, label_column, vocab, max_len=MAX_LEN):
        self.texts = df[text_column].tolist()
        self.labels = df[label_column].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def tokenize(self, text):
        tokens = [self.vocab.get(word, 1) for word in text.split()]
        tokens = tokens[:self.max_len] + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        input_ids = self.tokenize(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label

# Criar datasets
train_dataset = FakeNewsDataset(df_train, "title", "fake_news", vocab)
test_dataset = FakeNewsDataset(df_test, "title", "fake_news", vocab)

# 🔹 Verificação extra para evitar erro de DataLoader vazio
if len(train_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("Erro: O dataset de treino ou teste está vazio!")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Dados preparados: {len(train_dataset)} treino, {len(test_dataset)} teste.")

# ** Modelo Transformer Melhorado **
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, num_classes=2, dropout=DROPOUT):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_LEN, embed_dim))

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ** Inicializar modelo, otimizador e loss **
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(len(vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ** Função de Treinamento e Avaliação Melhorada **
def train_model(model, train_loader, optimizer, criterion, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0

        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    print("✅ Treinamento Concluído!")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Accuracy:", accuracy_score(all_labels, all_preds))

# ** Executar treinamento e avaliação **
train_model(model, train_loader, optimizer, criterion, epochs=EPOCHS)
evaluate_model(model, test_loader)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS
}
torch.save(checkpoint, "custom_transformer_model_v4.pth")

