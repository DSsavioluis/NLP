import pandas as pd
import re
import torch
from collections import defaultdict

class TextEncoder:
    def __init__(self, dataset_path, text_columns):
        """
        Inicializa a classe TextEncoder com o caminho do dataset e as colunas textuais.

        :param dataset_path: Caminho para o arquivo do dataset.
        :param text_columns: Lista com os nomes das colunas textuais do dataset.
        """
        self.dataset_path = dataset_path
        self.text_columns = text_columns
        self.df = None
        self.word2idx = None
        self.max_len = 0  # Guardar o tamanho máximo da sequência para padding

    def load_data(self):
        """Carrega o dataset do caminho especificado."""
        self.df = pd.read_csv(self.dataset_path)
        print("✅ Dataset carregado com sucesso!")

    def tokenize_text(self):
        """Realiza a tokenização manual das colunas textuais."""
        for col in self.text_columns:
            self.df[col] = self.df[col].astype(str).str.lower().apply(lambda x: re.findall(r'\b\w+\b', x))  # Tokenização
        print("✅ Tokenização realizada com sucesso!")

    def build_vocab(self):
        """Cria o vocabulário único a partir das colunas textuais tokenizadas."""
        vocab = {"<PAD>": 0, "<UNK>": 1}  # Incluindo tokens especiais
        idx = 2  # Começamos do 2 para evitar conflitos
        for col in self.text_columns:
            for sentence in self.df[col]:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
        self.word2idx = vocab
        print(f"✅ Vocabulário criado com {len(self.word2idx)} palavras.")

    def encode_text(self, sentence):
        """Codifica uma sentença substituindo as palavras pelos índices."""
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]

    def apply_encoding(self):
        """Aplica a codificação nas colunas textuais do dataset e define o tamanho máximo da sequência."""
        for col in self.text_columns:
            self.df[col + "_encoded"] = self.df[col].apply(self.encode_text)
            self.max_len = max(self.max_len, self.df[col + "_encoded"].apply(len).max())  # Atualiza max_len
        print("✅ Codificação aplicada com sucesso!")

    def apply_padding(self):
        """Aplica padding para todas as sequências garantindo que tenham o mesmo tamanho."""
        for col in self.text_columns:
            padded_col = col + "_padded"
            self.df[padded_col] = self.df[col + "_encoded"].apply(
                lambda x: x + [self.word2idx["<PAD>"]] * (self.max_len - len(x))
            )
        print("✅ Padding aplicado com sucesso!")

    def convert_to_tensor(self):
        """Converte as colunas encodificadas em tensores PyTorch."""
        for col in self.text_columns:
            tensor_col = col + "_tensor"
            self.df[tensor_col] = self.df[col + "_padded"].apply(lambda x: torch.tensor(x, dtype=torch.long))
        print("✅ Conversão para tensores realizada!")

    def save_encoded_data(self, output_path):
        """Salva o dataset com os dados encodificados e padding em um arquivo CSV."""
        self.df.to_csv(output_path, index=False)
        print(f"✅ Dataset encodificado salvo em {output_path}")


dataset_path = 'df_fake_news'
text_columns = ["id","title","fake_news"]

encoder = TextEncoder(dataset_path, text_columns)

encoder.load_data()
encoder.tokenize_text()
encoder.build_vocab()
encoder.apply_encoding()
encoder.apply_padding()
encoder.convert_to_tensor()
encoder.save_encoded_data("dataset_encoded.csv")
