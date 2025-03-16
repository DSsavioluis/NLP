import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import re

class TextDataset(Dataset):
    def __init__(self, dataframe, text_columns, label_column, word2idx, max_len):
        """
        Dataset para LLMs no PyTorch.

        :param dataframe: DataFrame contendo os dados encodificados.
        :param text_columns: Lista das colunas textuais a serem usadas.
        :param label_column: Nome da coluna que contém os rótulos (target).
        :param word2idx: Dicionário do vocabulário para conversão.
        :param max_len: Tamanho fixo das sequências após padding.
        """
        self.data = dataframe
        self.text_columns = text_columns
        self.label_column = label_column
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        """Retorna o número total de exemplos."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retorna um exemplo no formato (input_ids, label)."""
        row = self.data.iloc[idx]

        # Concatenar os textos das colunas especificadas
        input_text = []
        for col in self.text_columns:
            input_text.extend(row[col])  # Obtendo tokens encodificados

        # Aplicar padding se necessário
        input_text = input_text[:self.max_len] + [0] * (self.max_len - len(input_text))

        # Converter para tensor
        input_ids = torch.tensor(input_text, dtype=torch.long)
        label_value = row[self.label_column]

        # Se for string com lista, converte corretamente
        if isinstance(label_value, str) and label_value.startswith("["):
            label_value = eval(label_value)  # Converte string -> lista

        # Se ainda for uma lista, pega o primeiro valor
        if isinstance(label_value, list):
            label_value = label_value[0]

        # Converter para inteiro e depois para tensor
        label = torch.tensor(int(label_value), dtype=torch.long)

        return input_ids, label


def prepare_dataset(dataset_path, text_columns, label_column, output_train_path, output_test_path, split_ratio=0.8):
    """
    Função para preparar e salvar os datasets de treino e teste.

    :param dataset_path: Caminho do arquivo CSV original.
    :param text_columns: Lista das colunas textuais a serem usadas.
    :param label_column: Nome da coluna de rótulo.
    :param output_train_path: Caminho para salvar o dataset de treino.
    :param output_test_path: Caminho para salvar o dataset de teste.
    :param split_ratio: Proporção de treino (padrão: 80% treino, 20% teste).
    """
    df = pd.read_csv(dataset_path)

    # Função para converter corretamente os valores
    def safe_eval(x):
        """Avalia uma string contendo uma lista de tokens para PyTorch."""
        try:
            if isinstance(x, str):
                return eval(x) if x.startswith("[") else x
            return x
        except Exception as e:
            print(f"Erro ao converter: {x} -> {e}")
            return []

    # Aplicar a conversão
    for col in text_columns:
        df[col] = df[col].apply(safe_eval)

    # Construção do vocabulário
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for col in text_columns:
        for seq in df[col]:
            for word_id in seq:
                if word_id not in vocab:
                    vocab[word_id] = idx
                    idx += 1

    max_len = max(df[col].apply(len).max() for col in text_columns)  # Definir tamanho máximo de sequência

    # Adicionar metadados ao dataset
    df["max_len"] = max_len
    df["vocab_size"] = len(vocab)

    # Dividir dataset em treino/teste
    train_size = int(split_ratio * len(df))
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Salvar datasets processados
    df_train.to_csv(output_train_path, index=False)
    df_test.to_csv(output_test_path, index=False)
    print(f"✅ Datasets de treino e teste salvos em: {output_train_path}, {output_test_path}")


# ** Executando o Pré-Processamento **
dataset_path = "dataset_encoded.csv"  # Arquivo original
output_train_path = "train_dataset.csv"  # Arquivo de treino
output_test_path = "test_dataset.csv"  # Arquivo de teste
text_columns = ["title_encoded"]  # Coluna textual processada
label_column = "fake_news"  # Coluna alvo

prepare_dataset(dataset_path, text_columns, label_column, output_train_path, output_test_path)
