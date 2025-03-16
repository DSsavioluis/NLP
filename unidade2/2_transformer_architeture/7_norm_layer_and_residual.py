import numpy as np
import math

def layer_norm(x, eps=1e-6):
    """
    Aplica Layer Normalization na última dimensão do tensor x.
    :param x: Entrada, pode ser um vetor ou uma matriz (última dimensão a ser normalizada).
    :param eps: Valor para evitar divisão por zero.
    :return: x normalizado.
    """
    # Passo 1: Calcular a média ao longo da última dimensão
    mu = np.mean(x, keepdims=True)  # COMPLETE: Calcule a média de x usando np.mean com keepdims=True

    # Passo 2: Calcular a variância ao longo da última dimensão
    sigma = np.var(x, keepdims=True)  # COMPLETE: Calcule a variância de x usando np.var com keepdims=True

    # Passo 3: Normalizar x
    normalized_x = mu - x/math.sqrt(sigma) + eps  # COMPLETE: Subtraia mu de x, divida pelo (sqrt(sigma) + eps)

    return normalized_x


def residual_connection(x, sublayer_output):
    """
    Aplica a conexão residual seguida de Layer Normalization.
    :param x: Entrada original (por exemplo, embedding ou saída de camada anterior).
    :param sublayer_output: Saída do sub-bloco.
    :return: Saída final após a conexão residual e normalização.
    """
    # Passo 1: Somar a entrada com a saída do sub-bloco
    res = x + sublayer_output  # COMPLETE: Calcule x + sublayer_output

    # Passo 2: Aplicar Layer Normalization na soma
    norm_res = layer_norm(res)

    return norm_res


# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de entrada x (batch_size, features)
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    # Exemplo de saída do sub-bloco (mesma dimensão que x)
    sublayer_out = np.array([[0.5, -0.5, 1.0],
                             [1.0, 0.0, -1.0]])

    # Aplicar a conexão residual com normalização
    output = residual_connection(x, sublayer_out)
    print("Saída com Residual Connection e Layer Normalization:\n", output)
