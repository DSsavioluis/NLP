import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from atribute_extraction import FeatureExtraction
from preprocessing import apply_stemming, apply_lemmatization

# Carregar o dataset
dataset = pd.read_csv('C:/Users/Viana/PycharmProjects/savio/NLP/AV1/processed_news_dataset.csv')

# Função para treinar e avaliar o classificador
def train_and_evaluate(X, y, model=None):
    if model is None:
        model = LogisticRegression()

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report


# Configurar os dados e rótulos
y = dataset['fake_news']  # Label
extractor = FeatureExtraction(dataset)

# Configurar os dados e rótulos
y = dataset['fake_news']  # Label
extractor = FeatureExtraction(dataset)

# Resultados organizados
results = {}

# Caso a) Todas as formas de extração com e sem pré-processamento
# Sem pré-processamento
dataset_no_preprocess = dataset.copy()
dataset_no_preprocess['clean_title'] = dataset_no_preprocess['title']

forms_of_extraction = ['statistical_analysis','count_vectorizer', 'tfidf_vectorizer', 'cooccurrence_matrix', 'word2vec']


for form in forms_of_extraction:
    print(f"--- Avaliando {form} ---")

    # Sem pré-processamento
    extractor_no_pre = FeatureExtraction(dataset)
    if form == 'statistical_analysis':
        X_no_pre = extractor_no_pre.statistical_analysis().iloc[:, -3:]
    elif form == 'cooccurrence_matrix':
        X_no_pre = extractor_no_pre.cooccurrence_matrix(window_size=2, n_components=50)
    elif form == 'word2vec':
        X_no_pre = extractor_no_pre.word2vec()
    else:
        X_no_pre = getattr(extractor_no_pre, form)()

    # Ajustar o tamanho de X_no_pre para corresponder ao tamanho de y
    if X_no_pre.shape[0] > len(y):
        print(f"Tamanho de X_no_pre maior que y. Truncando X_no_pre de {X_no_pre.shape[0]} para {len(y)}.")
        X_no_pre = X_no_pre[:len(y)]

    y_no_pre = y.iloc[:X_no_pre.shape[0]]

    if X_no_pre.shape[0] == len(y_no_pre):
        acc_no_pre, report_no_pre = train_and_evaluate(X_no_pre, y_no_pre)
        results[f"{form}_no_pre"] = acc_no_pre
        print(report_no_pre)
    else:
        print(f"Inconsistência detectada no {form} sem pré-processamento! Tamanho de X_no_pre: {X_no_pre.shape[0]}, Tamanho de y: {len(y_no_pre)}")
        continue


    # Com pré-processamento
    print(f"--- Avaliando {form} com pré-processamento ---")

    dataset_pre = dataset.copy()
    dataset_pre['clean_title'] = dataset_pre['clean_title'].apply(apply_stemming)
    extractor_pre = FeatureExtraction(dataset_pre)

    if form == 'statistical_analysis':
        X_pre = extractor_pre.statistical_analysis().iloc[:, -3:]
    elif form == 'cooccurrence_matrix':
        X_pre = extractor_pre.cooccurrence_matrix(window_size=2, n_components=50)
    elif form == 'word2vec':
        X_pre = extractor_pre.word2vec()
    else:
        X_pre = getattr(extractor_pre, form)()

    # Verificação de inconsistência
    y_pre = y.iloc[:X_pre.shape[0]]
    if X_pre.shape[0] != len(y_pre):
        print(f"Inconsistência detectada no {form} com pré-processamento! Tamanho de X_pre: {X_pre.shape[0]}, Tamanho de y: {len(y_pre)}")
        continue

    acc_pre, report_pre = train_and_evaluate(X_pre, y_pre)
    results[f"{form}_pre"] = acc_pre
    print(report_pre)


# Organizar resultados do caso a)
print("\n--- Resultados do Caso a) ---")
print(pd.DataFrame(results.items(), columns=["Método", "Acurácia"]).sort_values(by="Acurácia", ascending=False))

# Caso b) Comparar formas de extração com pré-processamento
print("\n--- Caso b): Comparando formas de extração ---")
best_form = max(results, key=results.get).replace('_no_pre', '').replace('_pre', '').replace('_no', '')
print(f"A melhor forma de extração foi: {best_form}")

# Caso c) Comparar lemmatização e stemming
dataset_stemming = dataset.copy()
dataset_stemming['clean_title'] = dataset_stemming['clean_title'].apply(apply_stemming)

dataset_lemmatization = dataset.copy()
dataset_lemmatization['clean_title'] = dataset_lemmatization['clean_title'].apply(apply_lemmatization)

# Extração usando a melhor forma
extractor_stemming = FeatureExtraction(dataset_stemming)
X_stemming = getattr(extractor_stemming, best_form)()

extractor_lemmatization = FeatureExtraction(dataset_lemmatization)
X_lemmatization = getattr(extractor_lemmatization, best_form)()

# Avaliação
acc_stemming, _ = train_and_evaluate(X_stemming, y)
acc_lemmatization, _ = train_and_evaluate(X_lemmatization, y)

print("\n--- Caso c): Comparando Stemming e Lemmatização ---")
print(f"Acurácia com Stemming: {acc_stemming}")
print(f"Acurácia com Lemmatização: {acc_lemmatization}")