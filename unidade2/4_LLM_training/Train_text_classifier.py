import torch
from torch.utils.data import DataLoader
from Dataset import SentimentDataset
from transformer_Encoder import build_vocab
import torch.nn as nn
import torch.optim as optim
import json

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Cria um tensor (max_len, d_model) com as posições
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x tem shape (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 4. Modelo Transformer customizado para classificação de sentimentos
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2,
                 dim_feedforward=512, num_classes=2, max_len=100, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # Usamos o TransformerEncoder do PyTorch
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (batch_size, seq_len)
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        # Adiciona Positional Encoding: (batch_size, seq_len, d_model)
        embedded = self.pos_encoder(embedded)
        # O Transformer do PyTorch espera entrada em (seq_len, batch_size, d_model)
        embedded = embedded.transpose(0, 1)
        # Passa pelo encoder Transformer
        transformer_out = self.transformer_encoder(embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # Aplicamos pooling (média ao longo da dimensão seq_len)
        pooled = transformer_out.mean(dim=0)  # shape: (batch_size, d_model)
        logits = self.fc_out(pooled)
        return logits


texts = [
    "a descrição era completamente falsa não gostei",
    "infelizmente o produto chegou quebrado e fora do prazo",
    "muito bom chegou antes do prazo e em perfeitas condições",
    "péssimo atendimento não recomendo essa loja",
    "produto de excelente qualidade recomendo a todos",
    "a embalagem estava rasgada mas o produto estava bom",
    "fiquei decepcionado não atendeu as minhas expectativas",
    "superou as expectativas comprarei novamente",
    "chegou no prazo mas a qualidade deixou a desejar",
    "entrega rápida e produto de ótima qualidade"
]
# Labels: 0 para negativo, 1 para positivo
labels = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]

# Constrói o vocabulário
vocab = build_vocab(texts, min_freq=1)
vocab_size = len(vocab)
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("Tamanho do vocabulário:", vocab_size)

# Cria o dataset e o dataloader
max_len = 40
dataset = SentimentDataset(texts, labels, vocab, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

# Cria o modelo
model = TransformerClassifier(vocab_size=vocab_size, d_model=128, nhead=4, num_encoder_layers=2,
                              dim_feedforward=512, num_classes=2, max_len=max_len, dropout=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. Loop de treinamento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch  # inputs: (batch_size, max_len), targets: (batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model.forward(inputs)  # outputs: (batch_size, num_classes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Salvar modelo treinado
torch.save(model.state_dict(), "transformer_model_classification.pth")