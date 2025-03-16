import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

nlp = spacy.load('pt_core_news_sm')
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()


def process_text(text):
    doc = nlp(text)
    corrected_text = " ".join([token.text for token in doc])
    tokens = word_tokenize(corrected_text, language='portuguese')
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]

    data = []
    for token in filtered_tokens:
        stem = stemmer.stem(token)
        lemma = nlp(token)[0].lemma_
        data.append([token, stem, lemma])

    df = pd.DataFrame(data, columns=["Token", "Stemming", "Lemmatization"])
    return df


text = "Os menino estão estudando muito no escola. Eles quer aprender."
df = process_text(text)
print(df)