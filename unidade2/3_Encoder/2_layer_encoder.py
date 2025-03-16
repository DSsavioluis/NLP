import torch
import torch.nn as nn
import time

# ---------------------- IMPLEMENTAÇÃO DO ENCODER TRANSFORMER ----------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Criar camada de embedding para representar tokens como vetores
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Criar um bloco único de Encoder usando PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        # Empilhar múltiplos blocos de Encoder (INCOMPLETO)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        return self.encoder(x)

# ---------------------- FUNÇÃO PARA MEDIR TEMPO DE EXECUÇÃO ----------------------
def medir_tempo_execucao(num_layers, input_tokens, d_model, num_heads, d_ff, vocab_size):
    """
    Mede o tempo de execução do Encoder Transformer para diferentes quantidades de camadas.
    """
    # Criar o modelo TransformerEncoder com num_layers (INCOMPLETO)
    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, vocab_size)

    # Medir tempo inicial
    inicio = time.time()

    # Passar os tokens pelo encoder
    _ = encoder(input_tokens)

    # Medir tempo final
    fim = time.time()

    return fim - inicio  # Tempo total de execução

# ---------------------- TESTE FINAL ----------------------
if __name__ == "__main__":
    # Parâmetros do modelo
    d_model = 8    # Dimensão do embedding
    num_heads = 2  # Número de cabeças de atenção
    d_ff = 16      # Dimensão da Feed-Forward Network
    vocab_size = 50
    seq_len = 10  # Tamanho da sequência
    batch_size = 1

    # Criando tokens de entrada aleatórios
    input_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Lista de diferentes quantidades de camadas para testar
    num_layers_list = [1, 2, 4, 6, 8, 12]

    # Medir e exibir tempo de execução para cada número de camadas
    for num_layers in num_layers_list:
        tempo = medir_tempo_execucao(num_layers, input_tokens, d_model, num_heads, d_ff, vocab_size)
        print(f"Camadas: {num_layers} | Tempo de Execução: {tempo:.4f} segundos")
