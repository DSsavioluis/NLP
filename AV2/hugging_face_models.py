import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Escolher um modelo pré-treinado da Hugging Face
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"

# Carregar tokenizador e modelo pré-treinado
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_auth_token=True)

# Verifica se há GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Função para converter o dataset encodificado para o formato Hugging Face
class HuggingFaceDataset(Dataset):
    def __init__(self, df, text_column, label_column):
        self.texts = df[text_column].tolist()  # Agora pega o texto puro!
        self.labels = df[label_column].apply(lambda x: int(x) if isinstance(x, (int, float)) else int(ast.literal_eval(x)[0])).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retorna um item formatado para entrada no Hugging Face
        """
        text = " ".join(map(str, self.texts[idx]))  # Junta os tokens em um texto
        label = self.labels[idx]

        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ** Carregar o dataset de teste processado **
dataset_path = "dataset_preprocessed.csv"
df = pd.read_csv(dataset_path)

df["title"] = df["title"].str.lower()

# Criar dataset para Hugging Face
test_dataset = HuggingFaceDataset(df, "title", "fake_news")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Realizar a predição no conjunto de teste
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)  # Obtém a classe prevista

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Gerar Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Gerar Matriz de Confusão
conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()