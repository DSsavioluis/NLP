import torch
import torch.nn as nn

# ---------------------- TOKENIZAÇÃO E VOCABULÁRIO ----------------------
def tokenize(text):
    """
    Tokeniza um texto e retorna uma lista de palavras únicas em ordem.
    """
    text = text.lower().replace(",", "").replace(".", "")  # Remover pontuações
    tokens = text.split()  # Separar por espaço
    return tokens

# Função para converter um texto em índices numéricos
def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

# ---------------------- IMPLEMENTAÇÃO DO ENCODER TRANSFORMER ----------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Criar camada de embedding para representar tokens como vetores (INCOMPLETO)
        self.embedding = nn.Embedding(vocab_size, d_model,padding_idx=0)

        # Criar um bloco único de Encoder usando PyTorch (INCOMPLETO)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        # Empilhar múltiplos blocos de Encoder (INCOMPLETO)
        self.encoder = nn.TransformerEncoder( encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)  # Aplicar a camada de embedding (INCOMPLETO)
        return self.encoder(x)# COMPLETE AQUI   # Passar pelo encoder

# ---------------------- TESTE FINAL ----------------------
if __name__ == "__main__":
    # Criando um vocabulário único a partir do texto
    text = "O Transformer é um modelo de aprendizado profundo muito eficiente para NLP"
    tokens = tokenize(text)
    vocab = {word: idx for idx, word in enumerate(tokens, start=1)}

    # Adicionando tokens especiais
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab) + 1  # Token para palavras desconhecidas

    # Converter o texto original para índices numéricos (INCOMPLETO)
    input_indices = torch.tensor(text_to_indices(text, vocab)).unsqueeze(0)  # (1, seq_len)

    # Parâmetros do modelo
    d_model = 8    # Dimensão do embedding
    num_heads = 2  # Número de cabeças de atenção
    d_ff = 16      # Dimensão da Feed-Forward Network
    num_layers = 2 # Número de camadas do Encoder
    vocab_size = len(vocab)

    # Criando o Encoder (INCOMPLETO)
    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, vocab_size)

    # Passando os tokens pelo encoder (INCOMPLETO)
    output = encoder(input_indices)

    # Impressão dos resultados
    print("\nTexto de Entrada:", text)
    print("Tokens:", input_indices)
    print("\nSaída do Encoder (Shape):", output.shape)
    print("Saída do Encoder (Valores):\n", output)
