import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Inicializa o bloco de Multi-Head Attention.
        :param d_model: Dimensão do embedding (modelo).
        :param num_heads: Número de cabeças de atenção.
        """
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimensão de cada cabeça

        # Matrizes de projeção para Q, K, V
        self.W_q = np.random.rand(d_model, d_model)  # Projeção para Q
        self.W_k = np.random.rand(d_model, d_model)  # Projeção para K
        self.W_v = np.random.rand(d_model, d_model)  # Projeção para V

        # Biases para Q, K, V
        self.b_q = np.random.rand(d_model)
        self.b_k = np.random.rand(d_model)
        self.b_v = np.random.rand(d_model)

        # Matriz de projeção final (após concatenação)
        self.W_o = np.random.rand(d_model, d_model)  # Projeção de saída
        self.b_o = np.random.rand(d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Calcula a atenção escalonada por produto escalar.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: A saída da atenção e os pesos de atenção.
        """
        # **Complete o cálculo da atenção**
        matmul_qk = np.matmul(Q, K.transpose(0, 2, 1)) # Produto escalar entre Q e K^T
        scaled_attention_logits = matmul_qk / np.sqrt(self.d_k)  # Escalonar os logits pelo tamanho de d_k
        attention_weights = self.softmax(scaled_attention_logits)  # Aplicar softmax nos logits escalonados
        output = np.matmul(attention_weights, V) # Multiplicar os pesos de atenção pela matriz V

        return output, attention_weights

    def split_heads(self, X):
        """
        Divide as matrizes Q, K, e V em múltiplas cabeças.
        :param X: Matriz a ser dividida (Q, K ou V).
        :return: Matriz reformatada para múltiplas cabeças.
        """
        # **Complete a divisão da matriz em múltiplas cabeças**
        batch_size, seq_len, d_model = X.shape
        X = np.reshape(X, (batch_size, seq_len, self.num_heads, self.d_k))  # Redimensionar para (batch_size, seq_len, num_heads, d_k)
        return np.transpose(X, (0, 2, 1, 3))  # Reorganizar os eixos para (batch_size, num_heads, seq_len, d_k)

    def forward(self, Q, K, V):
        """
        Executa o processo de Multi-Head Attention.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: Saída do bloco de Multi-Head Attention.
        """
        # Passo 1: Aplicar as camadas lineares para projetar Q, K, V
        Q_proj = np.matmul(Q, self.W_q) + self.b_q  # Projeção de Q
        K_proj = np.matmul(K, self.W_k) + self.b_k  # Projeção de K
        V_proj = np.matmul(V, self.W_v) + self.b_v  # Projeção de V

        # Passo 2: Dividir em múltiplas cabeças
        Q_heads = self.split_heads(Q_proj)   # Dividir Q_proj em cabeças
        K_heads = self.split_heads(K_proj)  # Dividir K_proj em cabeças
        V_heads = self.split_heads(V_proj) # Dividir V_proj em cabeças

        # Passo 3: Aplicar atenção em cada cabeça
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]  # Seleciona a cabeça i
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Passo 4: Concatenar as saídas de todas as cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)  # Concatenar as saídas das cabeças

        # Passo 5: Aplicar a camada linear final
        output = np.matmul(concatenated, self.W_o) + self.b_o # Projeção final após concatenar as cabeças

        return output

    def softmax(self, x):
        # Passo 1: Subtrair o valor máximo de x para estabilidade numérica
        e_x = x - np.max(x)

        # Passo 2: Calcular o exponencial de cada elemento de x
        expoente = np.exp(e_x)

        # Passo 3: Normalizar os valores exponenciais, dividindo cada valor pelo somatório dos exponenciais
        return expoente / np.sum(expoente, axis=1, keepdims=True)

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)

    # Parâmetros do modelo
    d_model = 6  # Dimensão do modelo (embedding)
    num_heads = 3  # Número de cabeças de atenção
    seq_len = 4  # Comprimento da sequência
    batch_size = 1  # Tamanho do batch

    # Matrizes de entrada (Q, K, V)
    Q = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de consultas
    K = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de chaves
    V = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de valores

    # Criar o bloco de Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Executar o forward pass
    output = mha.forward(Q, K, V)
    print("Saída do Multi-Head Attention:\n", output)
