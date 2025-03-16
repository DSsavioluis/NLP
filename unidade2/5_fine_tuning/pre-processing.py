from datasets import Dataset
from transformers import AutoTokenizer
from src.utils import load_movie_review_clean_dataset

# Carregar e dividir o dataset
data = load_movie_review_clean_dataset()
data = {"text": data['text'].values, "label": data['label'].values}
dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

# Configurar tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format("torch")

# Salvar tokenizer e dataset pré-processado
tokenizer.save_pretrained("preprocessed_data")
import torch
torch.save(tokenized_datasets, "preprocessed_data/dataset.pt")