import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Caminhos dos modelos treinados
FINE_TUNED_MODEL_PATH = "./fake_news_model"  # Modelo fine-tuned (questão 4)
ENCODER_MODEL_PATH = "custom_transformer_model_v3.pth"  # Modelo encoder próprio (questão 5)

"custom_transformer_model_v3.pth = 51%"

# Carregar dataset de teste
df_test = pd.read_csv("dataset_preprocessed.csv")

# Converter labels para inteiros corretamente
df_test["fake_news"] = df_test["fake_news"].apply(
    lambda x: int(eval(x)[0]) if isinstance(x, str) and x.startswith("[") else int(x))

# Carregar tokenizer (mesmo usado no treinamento)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Criar dataset PyTorch
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, df, text_column, label_column, tokenizer):
        self.texts = df[text_column].tolist()
        self.labels = df[label_column].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=128,
                                  return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Criar dataset de teste
test_dataset = FakeNewsDataset(df_test, "title", "fake_news", tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 1️⃣ Avaliação do Modelo Fine-Tuned
print("\n🔍 Avaliação do Modelo Fine-Tuned")

# Carregar modelo fine-tuned
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
fine_tuned_model.eval()

# Predições
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = fine_tuned_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Gerar Classification Report e Matriz de Confusão
print("\nClassification Report - Fine-Tuned Model:")
print(classification_report(all_labels, all_preds, digits=4))

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Fine-Tuned Model")
plt.show()

# 2️⃣ Avaliação do Modelo Encoder Próprio
print("\n🔍 Avaliação do Modelo Encoder Próprio")

# Recriar arquitetura exata usada no treinamento
class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_heads=8, hidden_dim=512, num_layers=6, output_dim=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, 128, embedding_dim))

        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                          dim_feedforward=hidden_dim)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Carregar o checkpoint salvo
checkpoint = torch.load(ENCODER_MODEL_PATH, map_location=torch.device("cpu"))

# Verificar se "model_state_dict" está no checkpoint e extraí-lo
if "model_state_dict" in checkpoint:
    print("\n🔍 Extraindo o state_dict do checkpoint salvo...")
    checkpoint = checkpoint["model_state_dict"]

# Listar todas as chaves disponíveis no state_dict
print("\n🔍 Chaves disponíveis no state_dict salvo:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Tentar encontrar a chave correta para o embedding e definir o vocab_size
vocab_size = None
for key in checkpoint.keys():
    if "embedding" in key:
        vocab_size = checkpoint[key].shape[0]
        print(f"\n📌 Tamanho do vocabulário salvo no modelo: {vocab_size}")
        break

# Se não encontrar um vocabulário, exibir erro
if vocab_size is None:
    raise KeyError("\n❌ Nenhuma chave relacionada a 'embedding' foi encontrada no state_dict. Verifique como o modelo foi salvo.")

# Criar modelo com o tamanho de vocabulário correto
encoder_model = TransformerClassifier(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=256,
    num_layers=3,
    output_dim=2
)

# Carregar pesos do modelo
encoder_model.load_state_dict(checkpoint)
encoder_model.eval()

# Predições
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Ajustar input_ids para não ultrapassar os índices do embedding
        input_ids = torch.clamp(input_ids, min=0, max=encoder_model.embedding.num_embeddings - 1)

        outputs = encoder_model(input_ids)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Gerar Classification Report e Matriz de Confusão
print("\nClassification Report - Encoder Model:")
print(classification_report(all_labels, all_preds, digits=4))

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Encoder Model")
plt.show()