import numpy as np
from collections import Counter

import pandas as pd


def calculate_tf_idf_table(corpus):

    # Tokenizer
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_corpus = [[word.lower() for word in doc] for doc in tokenized_corpus]
    unique_terms = sorted(set(term for doc in tokenized_corpus for term in doc))
    term_index = {term: idx for idx, term in enumerate(unique_terms)}

    # Term Frequency (TF)
    tf_matrix = np.zeros((len(corpus), len(unique_terms)))
    for doc_idx, doc in enumerate(tokenized_corpus):
        term_counts = Counter(doc)
        for term, count in term_counts.items():
            tf_matrix[doc_idx, term_index[term]] = count / len(doc)

    # Inverse Document Frequency (IDF)
    doc_count = len(corpus)
    idf_vector = np.zeros(len(unique_terms))
    for term, idx in term_index.items():
        doc_freq = sum(1 for doc in tokenized_corpus if term in doc)
        idf_vector[idx] = np.log((doc_count + 1) / (doc_freq + 1)) + 1  # Adiciona 1 para suavizar

    # TF-IDF
    tf_idf_matrix = tf_matrix * idf_vector
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=unique_terms, index=corpus)
    return tf_idf_df


if __name__ == "__main__":
    corpus = [
        "O gato roeu a roupa do rei de roma",
        "O rei roeu o rato de roma"
    ]
    tf_idf_table = calculate_tf_idf_table(corpus)
    print("Matriz TF-IDF (Formato Tabela):")
    print(tf_idf_table)