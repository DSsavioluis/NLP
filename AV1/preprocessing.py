import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Baixar recursos necessários para nltk
'''nltk.download('stopwords')
nltk.download('wordnet')'''

# Carregar os datasets
news_fake = pd.read_csv("News_fake.csv")
news_not_fake = pd.read_csv("News_notFake.csv")

# Adicionar a coluna 'fake_news' com valores booleanos
news_fake['fake_news'] = 1
news_not_fake['fake_news'] = 0

# Concatenar os dois datasets
combined_news = pd.concat([news_fake, news_not_fake], ignore_index=True)

combined_news = combined_news.drop(columns=['tweet_ids', 'news_url'])

# Instâncias para stemming e lematização
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Lista de stopwords
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text: str) -> str:
    # Remover URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remover menções (@usuário)
    text = re.sub(r'@\w+', '', text)
    # Converter para minúsculas
    text = text.lower()
    # Remover caracteres especiais
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def apply_stemming(text: str) -> str:
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def apply_lemmatization(text: str) -> str:
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def process_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if 'title' in dataset.columns:
        dataset['title'] = dataset['title'].astype(str)
        dataset['clean_title'] = dataset['title'].apply(preprocess_text)
        dataset['clean_title'] = dataset['clean_title'].apply(remove_stopwords)
        dataset['stemmed_title'] = dataset['clean_title'].apply(apply_stemming)
        dataset['lemmatized_title'] = dataset['clean_title'].apply(apply_lemmatization)
    return dataset

#Aplicanodo as funções
processed_news = process_dataset(combined_news)

print(processed_news)
# Salvar o dataset processado
processed_news.to_csv('processed_news_dataset.csv',index=False)

