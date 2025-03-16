import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score

# Modelo pré-treinado
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"

# Carregar tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # 2 classes (Fake/Real)

# Carregar dataset processado
df = pd.read_csv("dataset_preprocessed.csv")

# Separar em treino (80%) e teste (20%)
train_size = int(0.8 * len(df))
df_train = df[:train_size]
df_test = df[train_size:]

# Dataset PyTorch para Fine-Tuning
class FakeNewsDataset(Dataset):
    def __init__(self, df, text_column, label_column):
        self.texts = df[text_column].tolist()
        self.labels = df[label_column].apply(lambda x: int(eval(x)[0]) if isinstance(x, str) and x.startswith("[") else int(x)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Criar datasets de treino e teste
train_dataset = FakeNewsDataset(df_train, "title", "fake_news")
test_dataset = FakeNewsDataset(df_test, "title", "fake_news")

# Função para calcular métricas (Acurácia)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


# Configurações de Treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

print(f"Quantidade de exemplos no dataset de treino: {len(train_dataset)}")
print(f"Quantidade de exemplos no dataset de teste: {len(test_dataset)}")

# Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Treinar o modelo
trainer.train()

# Salvar modelo fine-tuned
model.save_pretrained("./fake_news_model")
tokenizer.save_pretrained("./fake_news_model")

print("✅ Modelo treinado e salvo com sucesso!")
