import numpy as np


def relu(z):
    """
    Implementa a função de ativação ReLU.
    :param z: Entrada (vetor ou matriz).
    :return: z transformado pela ReLU.
    """
    # COMPLETE AQUI: Retorne o máximo entre 0 e z
    return np.maximum(0, z)


def feed_forward(x, W1, b1, W2, b2):
    """
    Implementa a camada Feed-Forward do Transformer.
    :param x: Entrada (batch_size, seq_len, d_model)
    :param W1: Matriz de pesos da primeira projeção (d_model, d_ff)
    :param b1: Bias da primeira projeção (d_ff,)
    :param W2: Matriz de pesos da segunda projeção (d_ff, d_model)
    :param b2: Bias da segunda projeção (d_model,)
    :return: Saída da Feed-Forward Network (batch_size, seq_len, d_model)
    """
    # Passo 1: Aplicar a primeira transformação linear W1 * x + b1
    z = np.dot(x, W1) + b1

    # Passo 2: Aplicar a ativação ReLU
    h = relu(z)

    # Passo 3: Aplicar a segunda transformação linear W2 * h + b2
    output1 = np.dot(h, W2) + b2

    return output1


# Teste da implementação
if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 2  # Número de frases no batch
    seq_len = 4  # Número de tokens por frase
    d_model = 6  # Dimensão do embedding
    d_ff = 12  # Dimensão oculta da FFN

    # Criando uma entrada aleatória (batch_size, seq_len, d_model)
    x = np.random.rand(batch_size, seq_len, d_model)

    # Criando pesos e bias aleatórios para a FFN
    W1 = np.random.rand(d_model, d_ff)  # Projeção para dimensão oculta
    b1 = np.random.rand(d_ff)  # Bias da primeira camada
    W2 = np.random.rand(d_ff, d_model)  # Projeção de volta para d_model
    b2 = np.random.rand(d_model)  # Bias da segunda camada

    # Aplicar Feed-Forward Network
    output = feed_forward(x, W1, b1, W2, b2)
    print("Saída da Feed-Forward Network:\n", output)
