import numpy as np


# Função para dividir as matrizes em múltiplas cabeças
def split_heads(X, num_heads):
    """
    Divide a matriz em múltiplas cabeças.
    :param X: Matriz original (batch_size, seq_len, d_model).
    :param num_heads: Número de cabeças.
    :return: Matriz reformatada (batch_size, num_heads, seq_len, d_k).
    """
    # Complete as operações abaixo para dividir X em múltiplas cabeças.
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads  # Dimensão de cada cabeça
    # Redimensione X para incluir o número de cabeças e a nova dimensão d_k
    X = X.reshape(batch_size, seq_len, num_heads, d_k)
    # **Complete aqui**

    # Reordene os eixos para (batch_size, num_heads, seq_len, d_k)
    X = np.transpose(X, (0, 2, 1, 3))
    # **Complete aqui**
    return X  # Retorne a matriz transposta dividida corretamente.


# Parâmetros
batch_size = 1  # Tamanho do lote
seq_len = 4  # Comprimento da sequência
d_model = 8  # Dimensão do modelo
num_heads = 2  # Número de cabeças
d_k = d_model // num_heads  # Dimensão de cada cabeça

# Matrizes de entrada (Q, K, V)
Q = np.random.rand(batch_size, seq_len, d_model)  # Consultas
K = np.random.rand(batch_size, seq_len, d_model)  # Chaves
V = np.random.rand(batch_size, seq_len, d_model)  # Valores

# Matrizes de pesos para Q, K, V
W_Q = np.random.rand(d_model, d_model)  # Pesos para Q
W_K = np.random.rand(d_model, d_model)  # Pesos para K
W_V = np.random.rand(d_model, d_model)  # Pesos para V

# Biases para Q, K, V
b_Q = np.random.rand(d_model)
b_K = np.random.rand(d_model)
b_V = np.random.rand(d_model)

# Passo 1: Aplicar as projeções lineares para Q, K, V
# **Complete o cálculo abaixo para Q_proj, K_proj, V_proj**
Q_proj = np.dot(Q, W_Q) + b_Q  # (batch_size, seq_len, d_model)
K_proj = np.dot(K, W_K) + b_K   # (batch_size, seq_len, d_model)
V_proj = np.dot(V, W_V) + b_V   # (batch_size, seq_len, d_model)

# Passo 2: Dividir as matrizes projetadas em múltiplas cabeças
# **Use a função split_heads para dividir Q_proj, K_proj e V_proj**
Q_heads = split_heads(Q_proj, num_heads) # (batch_size, num_heads, seq_len, d_k)
K_heads = split_heads(K_proj, num_heads)  # (batch_size, num_heads, seq_len, d_k)
V_heads = split_heads(V_proj, num_heads)  # (batch_size, num_heads, seq_len, d_k)

# Exibir dimensões e valores
print("Q_proj (após projeção linear):", Q_proj.shape if Q_proj is not None else "Incomplete")
print(Q_proj)
print("\nQ_heads (após divisão em cabeças):", Q_heads.shape if Q_heads is not None else "Incomplete")
print(Q_heads)

print("\nK_proj (após projeção linear):", K_proj.shape if K_proj is not None else "Incomplete")
print(K_proj)
print("\nK_heads (após divisão em cabeças):", K_heads.shape if K_heads is not None else "Incomplete")
print(K_heads)

print("\nV_proj (após projeção linear):", V_proj.shape if V_proj is not None else "Incomplete")
print(V_proj)
print("\nV_heads (após divisão em cabeças):", V_heads.shape if V_heads is not None else "Incomplete")
print(V_heads)
