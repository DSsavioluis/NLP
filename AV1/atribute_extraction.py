import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter
from gensim.models import Word2Vec
import numpy as np

dataset = pd.read_csv('C:/Users/Viana/PycharmProjects/savio/NLP/AV1/processed_news_dataset.csv')

class FeatureExtraction:
    def __init__(self, dataset: pd.DataFrame, text_column: str = "clean_title"):
        self.dataset = dataset
        self.text_column = text_column

    def statistical_analysis(self) -> pd.DataFrame:

        stats = self.dataset[self.text_column].apply(lambda x: {
            "word_count": len(x.split()),
            "avg_word_length": np.mean([len(word) for word in x.split()]),
            "char_count": len(x)
        })

        # Converter para DataFrame
        stats_df = pd.DataFrame(stats.tolist())
        self.dataset = pd.concat([self.dataset, stats_df], axis=1)
        return self.dataset

    def count_vectorizer(self, max_features=500):
        vectorizer = CountVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(self.dataset[self.text_column])
        feature_names = vectorizer.get_feature_names_out()

        return pd.DataFrame(X.toarray(), columns=feature_names)

    def tfidf_vectorizer(self, max_features=500):
        tfidf = TfidfVectorizer(max_features=max_features)
        X = tfidf.fit_transform(self.dataset[self.text_column])
        feature_names = tfidf.get_feature_names_out()

        return pd.DataFrame(X.toarray(), columns=feature_names)

    def cooccurrence_matrix(self, window_size=2, n_components=50):
        # Obter todas as palavras
        words = [text.split() for text in self.dataset[self.text_column]]

        # Criar vocabulário e matriz
        cooccurrence = Counter()
        for sentence in words:
            for i, word in enumerate(sentence):
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(sentence))
                for j in range(start, end):
                    if i != j:
                        cooccurrence[(word, sentence[j])] += 1

        vocab = list(set([word for sentence in words for word in sentence]))
        vocab_size = len(vocab)
        word_to_index = {word: i for i, word in enumerate(vocab)}

        # Matriz de coocorrência
        co_matrix = np.zeros((len(words), vocab_size))  # Tamanho alinhado com o número de textos
        for i, sentence in enumerate(words):
            for word in sentence:
                if word in word_to_index:
                    co_matrix[i, word_to_index[word]] += 1

        # Aplicar PCA
        if co_matrix.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            co_matrix = pca.fit_transform(co_matrix)

        return co_matrix

    def word2vec(self, vector_size=100, window=5, min_count=1, workers=4):
        sentences = [text.split() for text in self.dataset[self.text_column]]
        model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        word_vectors = model.wv
        vectors = []
        for sentence in sentences:
            sentence_vectors = [word_vectors[word] for word in sentence if word in word_vectors]
            if sentence_vectors:
                vectors.append(np.mean(sentence_vectors, axis=0))
            else:
                vectors.append(np.zeros(vector_size))

        return np.array(vectors)


# Instanciar a classe
extractor = FeatureExtraction(dataset)

# Análise estatística
dataset_stats = extractor.statistical_analysis()
print("Análise Estatística:\n", dataset_stats.head())

# CountVectorizer
count_matrix = extractor.count_vectorizer()
print("Count Vectorizer:\n", count_matrix.head())

# TF-IDF
tfidf_matrix = extractor.tfidf_vectorizer()
print("TF-IDF:\n", tfidf_matrix.head())

# Gerar matriz de coocorrência reduzida
cooccurrence_reduced = extractor.cooccurrence_matrix(window_size=2, n_components=50)
print("Matriz de Coocorrência Reduzida:")
print(cooccurrence_reduced)

# Word2Vec
word_vectors = extractor.word2vec()
print("Exemplo de vetores Word2Vec (primeiros 5 textos):")
for i, vector in enumerate(word_vectors[:5]):
    print(f"Texto {i+1}: {vector[:5]}")

